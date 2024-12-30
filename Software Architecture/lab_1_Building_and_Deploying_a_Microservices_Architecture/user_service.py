from fastapi import FastAPI

app = FastAPI()

# Sample user data
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
]

@app.get("/users")
def get_users():
    """Retrieve a list of users."""
    return {"users": users}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    """Retrieve details of a specific user."""
    user = next((u for u in users if u["id"] == user_id), None)
    if user:
        return user
    return {"error": "User not found"}, 404
