from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from flashrank import Ranker, RerankRequest
from pydantic import BaseModel
from typing import List, Optional, Union
import json
import logging
import uuid

app = FastAPI()

# Nano (~4MB), blazing fast model & competitive performance (ranking precision).
ranker = Ranker()

# Small (~34MB), slightly slower & best performance (ranking precision).
# ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# Medium (~110MB), slower model with best zeroshot performance (ranking precision) on out of domain data.
# ranker = Ranker(model_name="rank-T5-flan")

# Setup logging
logging.basicConfig(level=logging.INFO)

class Document(BaseModel):
    text: str
    meta: Optional[dict] = None

class RankRequest(BaseModel):
    model: str
    query: str
    documents: List[Union[Document, str]]
    top_n: Optional[int] = None
    rank_fields: Optional[List[str]] = None
    return_documents: Optional[bool] = True
    max_chunks_per_doc: Optional[int] = None

class RankResponseMeta(BaseModel):
    api_version: dict
    billed_units: dict
    tokens: dict
    warnings: List[str]

class RankResponseItem(BaseModel):
    document: Optional[Document] = None
    index: int
    relevance_score: float

class RankResponse(BaseModel):
    id: str
    results: List[RankResponseItem]
    meta: RankResponseMeta

def generate_response(request: RankRequest):
    passages_dict = []
    for idx, doc in enumerate(request.documents):
        if isinstance(doc, str):
            passages_dict.append({"id": str(idx), "text": doc, "meta": {}})
        else:
            passages_dict.append({"id": str(idx), "text": doc.text, "meta": doc.meta or {}})

    rerankrequest = RerankRequest(query=request.query, passages=passages_dict)
    result = ranker.rerank(rerankrequest)

    final_result = []
    for item in result:
        final_result.append({
            "document": {"text": item.get("text"), "meta": item.get("meta")},
            "index": int(item.get("id")),
            "relevance_score": float(item.get("score"))
        })

    if request.top_n:
        final_result = final_result[:request.top_n]

    if not request.return_documents:
        for res in final_result:
            res.pop("document")

    response_id = str(uuid.uuid4())

    response = RankResponse(
        id=response_id,
        results=final_result,
        meta=RankResponseMeta(
            api_version={"version": "1.0", "is_deprecated": False, "is_experimental": False},
            billed_units={"input_tokens": 0, "output_tokens": 0, "search_units": 0, "classifications": 0},
            tokens={"input_tokens": 0, "output_tokens": 0},
            warnings=[]
        )
    )

    return response

@app.post("/v1/rerank")
@app.post("/rerank")
async def rerank(request: RankRequest, x_client_name: str = Header(None)):
    logging.info(f"Received request: {request.json()}")
    response = generate_response(request)
    response_json = response.json()
    logging.info(f"Response JSON: {response_json}")
    return JSONResponse(content=json.loads(response_json))

# To run the app, use the command: uvicorn my_module:app --reload
# To run the app on network, use the command: uvicorn my_module:app --host 0.0.0.0 --reload
