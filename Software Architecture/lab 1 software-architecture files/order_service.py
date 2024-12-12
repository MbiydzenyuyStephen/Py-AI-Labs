from fastapi import FastAPI
import requests

app = FastAPI()

# Simulated database for orders
orders = []

# URL for the Product Service
PRODUCT_SERVICE_URL = "http://product-service.default.svc.cluster.local:80/products"

@app.post("/orders")
def create_order(product_id: int, quantity: int):
    """Create an order for a product."""
    # Fetch product details from Product Service
    response = requests.get(f"{PRODUCT_SERVICE_URL}/{product_id}")
    if response.status_code != 200:
        return {"error": "Product not found"}, 404

    product = response.json()
    total_price = product["price"] * quantity
    order = {"product_id": product_id, "quantity": quantity, "total_price": total_price}
    orders.append(order)

    return {"message": "Order created successfully", "order": order}

@app.get("/orders")
def get_orders():
    """Retrieve all orders."""
    return {"orders": orders}
