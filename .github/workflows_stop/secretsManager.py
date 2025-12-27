import os
import json
import boto3
import sys
from botocore.exceptions import ClientError

# Configuration
REGION = os.getenv("AWS_REGION")
ACTION = os.getenv("ACTION_TYPE")
SECRET_NAME = os.getenv("SECRET_FILE_NAME")
TARGET_REPO = os.getenv("REPO_NAME") 
SERVICES_JSON = os.getenv("PROVISIONED_MAP", "{}")

# Constants
CREATE_FILE_STRUCTURE = "CREATE_FILE_STRUCTURE"
SERVICE_REQUEST = "SERVICE_REQUEST"
ADD_SERVICES = "ADD_SERVICES"
CHECK_MAPPING = "CHECK_MAPPING"

def get_client():
    return boto3.client("secretsmanager", region_name=REGION)

def handle_secret():
    client = get_client()
    
    try:
        # 1. Fetch Existing Secret
        full_data = {
            "proxy": "http://127.0.0.1/",
            "max_port_used": "7999",
            "repo_mapping": {},
            "repos": {}
        }
        
        try:
            response = client.get_secret_value(SecretId=SECRET_NAME)
            if 'SecretString' in response:
                full_data.update(json.loads(response['SecretString']))
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"Secret {SECRET_NAME} not found. Creating new.")
            else:
                raise e

        # 2. Handler: CREATE_FILE_STRUCTURE
        if ACTION == CREATE_FILE_STRUCTURE:
            runner_host = os.getenv("SELF_HOSTED_RUNNER")
            if runner_host: full_data["runner_host"] = runner_host
            
            full_data.setdefault("repos", {})
            full_data.setdefault("repo_mapping", {})
            full_data.setdefault("max_port_used", "7999")
            full_data.setdefault("proxy", "http://127.0.0.1/")

            client.put_secret_value(SecretId=SECRET_NAME, SecretString=json.dumps(full_data))
            print(f"SUCCESS: Infrastructure secrets initialized.")

        # 3. Handler: SERVICE_REQUEST
        elif ACTION == SERVICE_REQUEST:
            requested_env = os.getenv("SERVICES", "")
            requested_list = [s.strip() for s in requested_env.split(";") if s.strip()]
            
            repo_services = full_data.get("repos", {}).get(TARGET_REPO, {}).get("services", {})
            result_map = {}
            needs_provisioning = False

            for req in requested_list:
                val = repo_services.get(req, "")
                result_map[req] = val
                if not val: needs_provisioning = True
            
            output_file = os.getenv('GITHUB_OUTPUT')
            if output_file:
                with open(output_file, "a") as f:
                    f.write(f"required_services={json.dumps(result_map)}\n")
                    f.write(f"needs_provisioning={'true' if needs_provisioning else 'false'}\n")
            print(f"Check Complete. Provisioning needed: {needs_provisioning}")

        # 4. Handler: ADD_SERVICES
        elif ACTION == ADD_SERVICES:
            try:
                tf_data = json.loads(SERVICES_JSON)
            except json.JSONDecodeError:
                print(f"ERROR: Invalid JSON: {SERVICES_JSON}")
                sys.exit(1)
            
            target_key = TARGET_REPO.lower() if TARGET_REPO else ""
            updates = {}

            if "s3_buckets" in tf_data and target_key in tf_data["s3_buckets"]:
                updates["S3"] = tf_data["s3_buckets"][target_key]
            if "ecr_repositories" in tf_data and target_key in tf_data["ecr_repositories"]:
                updates["ECR"] = tf_data["ecr_repositories"][target_key]

            # Update Structure
            repos = full_data.setdefault("repos", {})
            repo_node = repos.setdefault(TARGET_REPO, {})
            services_node = repo_node.setdefault("services", {})
            services_node.update(updates)
            
            client.put_secret_value(SecretId=SECRET_NAME, SecretString=json.dumps(full_data))
            print(f"SUCCESS: Updated {TARGET_REPO} with {updates}")

        # 5. Handler: CHECK_MAPPING
        elif ACTION == CHECK_MAPPING:
            repo_mapping = full_data.get("repo_mapping", {})
            is_mapped = False
            assigned_port = ""

            # A. Port Assignment
            if TARGET_REPO in repo_mapping:
                is_mapped = True
                assigned_port = repo_mapping[TARGET_REPO]
            else:
                try:
                    current_max = int(full_data.get("max_port_used", "7999"))
                except ValueError:
                    current_max = 7999
                
                next_port = current_max + 1
                assigned_port = str(next_port)
                
                full_data["repo_mapping"][TARGET_REPO] = assigned_port
                full_data["max_port_used"] = assigned_port
                
                client.put_secret_value(SecretId=SECRET_NAME, SecretString=json.dumps(full_data))
                print(f"Assigned new port {assigned_port} for {TARGET_REPO}")

            # B. Outputs
            base_proxy = full_data.get("proxy", "http://127.0.0.1/")
            clean_base = base_proxy.rstrip("/")
            proxy_target = f"{clean_base}:{assigned_port}/"

            # ECR Repo Name (Extract lowercase name)
            repo_services = full_data.get("repos", {}).get(TARGET_REPO, {}).get("services", {})
            raw_ecr_name = repo_services.get("ECR", TARGET_REPO)
            ecr_repo_name = raw_ecr_name.lower()

            # Write to GitHub Output
            output_file = os.getenv('GITHUB_OUTPUT')
            if output_file:
                with open(output_file, "a") as f:
                    f.write(f"is_mapped={'true' if is_mapped else 'false'}\n")
                    f.write(f"proxy_target={proxy_target}\n")
                    f.write(f"ecr_repo_name={ecr_repo_name}\n")

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    if not all([REGION, SECRET_NAME, ACTION]):
        print("CRITICAL: Missing required vars.")
        sys.exit(1)
    handle_secret()