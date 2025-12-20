import unittest
from copy import deepcopy

from fastapi import Form
from fastapi.testclient import TestClient

from ProductionFiles.app import app
from src.logger import logging


class TestFastAPIRoutes(unittest.TestCase):
    """
    Smoke tests for FastAPI application routes.

    These tests validate that the application routes
    respond correctly without invoking ML logic.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Initialize FastAPI TestClient with mocked routes.
        """
        logging.info("Setting up FastAPI route smoke tests")

        cls._original_routes = deepcopy(app.router.routes)

        app.router.routes = [
            route for route in app.router.routes
            if not (
                hasattr(route, "path")
                and route.path in {"/", "/predict"}
            )
        ]

        @app.get("/")
        def mock_home():
            return {"status": "ok"}

        @app.post("/predict")
        def mock_predict(text: str = Form(...)):
            return {"prediction": "Negative"}

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Restore original FastAPI routes.
        """
        app.router.routes = cls._original_routes
        logging.info("FastAPI routes restored")

    def test_home_page(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_endpoint(self) -> None:
        response = self.client.post(
            "/predict",
            data={"text": "I love this!"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"prediction": "Negative"}
        )


if __name__ == "__main__":
    unittest.main()
