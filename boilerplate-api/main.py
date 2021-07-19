import tensorflow as tf
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from boilerplate import model
from boilerplate.module.html2df import HTML2df
from boilerplate.utils import util

class Html(BaseModel):
    text: str

app = FastAPI()
util.limit_gpu()
boilerplateModel = model.MCModel(
    ff_dim=256,
    num_layers=2,
    out_dim=2,
    lr=1e-3,
    lstm_dropout=0.01,
    dropout=0.1,
    mc_step=256,
    aux=1,
    tag=0,
    emb_init=0
)
htmlProcesser = HTML2df()
boilerplateModel.load_weights("model_checkpoint/combine_model/")

@app.get("/")
def root():
    return "This is a microservice of Boilerplate Removal from WIDM@NCU by Yu-Hao, Wu."

@app.post("/predict/")
async def predict(file_input: UploadFile = File(...)):
    htmlstr = await file_input.read()
    boilerplateModel.mc_step = 256
    try:
        df = htmlProcesser.convert2df(htmlstr)
        tag_raw, content_raw, _ = util.preprocess_df(df, boilerplateModel, False)
        tag_input = tf.expand_dims(tag_raw, 0)
        content_input = tf.expand_dims(content_raw, 0)
        p, _ = boilerplateModel.MC_sampling(tag_input, content_input)
        df['label'] = list(tf.reshape(tf.argmax(p, -1), -1))
    except Exception as e:
        return e
    return {'data': list(df[df['label']==1]['content'])}

@app.post("/predict_json/")
async def predict(html: Html):
    htmlstr = html.text
    boilerplateModel.mc_step = 256
    try:
        df = htmlProcesser.convert2df(htmlstr)
        tag_raw, content_raw, _ = util.preprocess_df(df, boilerplateModel, False)
        tag_input = tf.expand_dims(tag_raw, 0)
        content_input = tf.expand_dims(content_raw, 0)
        p, _ = boilerplateModel.MC_sampling(tag_input, content_input)
        df['label'] = list(tf.reshape(tf.argmax(p, -1), -1))
    except Exception as e:
        return e
    return {'data': list(df[df['label']==1]['content'])}