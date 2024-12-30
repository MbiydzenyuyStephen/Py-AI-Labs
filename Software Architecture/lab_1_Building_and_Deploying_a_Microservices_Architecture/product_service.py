from fastapi import FastAPI

app = FastAPI()

# Sample product data
products = [
    {"id": 1, "name": "Laptop", "price": 1000},
    {"id": 2, "name": "Smartphone", "price": 500},
    {"id": 3, "name": "Tablet", "price": 300},
]

@app.get("/products")
def get_products():
    """Retrieve a list of products."""
    return {"products": products}

@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Retrieve details of a specific product."""
    product = next((p for p in products if p["id"] == product_id), None)
    if product:
        return product
    return {"error": "Product not found"}, 404
