from fastapi import Query
from typing import Optional, List

@app.get("/products")
def get_products(
    categorpy: Optional[str]=None,
    min_price: Optional[float]=None,
    max_price: Optional[float]=None,
    tags: List[str]=Query(default=[]),
    in_stock: bool=True
):
    filters={
        "category":category,
        "price_range":f""
    }