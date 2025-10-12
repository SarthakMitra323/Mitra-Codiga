# Test the web app output capture
from web_app import app
import json

# Test the API endpoint
with app.test_client() as client:
    # Test simple print
    response = client.post('/api/execute', 
                          json={'code': 'print("Hello, Web!")\nprint("Second line")'})
    
    print("Response status:", response.status_code)
    print("Response data:", response.get_json())
    
    # Test with variables
    response = client.post('/api/execute', 
                          json={'code': 'let x = 5\nprint("x =", x)\nlet y = x * 2\nprint("y =", y)'})
    
    print("\nVariable test:")
    print("Response data:", response.get_json())