

import tomllib
from pathlib import Path
from supabase import create_client

with open(".streamlit/secrets.toml", "rb") as file:
    secrets = tomllib.load(file)

supabase = create_client(
    secrets["SUPABASE_URL"],
    secrets["SUPABASE_KEY"]
)

response = (
    supabase.table("customers")
    .select("customer_id")
    .limit(1)
    .execute()
)

print("Response type:", type(response.data))
print("Response data:", repr(response.data))
print("Number of rows:", len(response.data))
