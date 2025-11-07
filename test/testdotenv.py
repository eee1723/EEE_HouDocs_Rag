import dotenv
import os
dotenv.load_dotenv()


a = os.getenv("A")
print("A=", type(a))