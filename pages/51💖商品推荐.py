import streamlit as st
import pandas as pd
from database.database import engine

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from common.product import display_products
from utils import init_page_header, init_session_state, get_avatar

title = "商品推荐"
icon = "💖"
init_page_header(title, icon)
init_session_state()


class CFDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df["user_id"].values)
        self.items = torch.LongTensor(df["product_id"].values)
        self.ratings = torch.FloatTensor(df["rating"].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx],self.items[idx],self.ratings[idx]


class CFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(CFModel, self).__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    
    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        ratings = (user_embedding * item_embedding).sum(dim=1)
        
        return ratings


def recommend_items(model, user_id, num_items):
    user_embedding = model.user_embeddings(torch.LongTensor([user_id - 1]))
    scores = torch.matmul(user_embedding, model.item_embeddings.weight.t())
    probabilities = torch.nn.functional.softmax(scores, dim=1)
    _, indices = torch.topk(probabilities, k=num_items, dim=1, largest=True, sorted=True)
    return indices[0].tolist()


# @st.cache_resource
def predict_model():
    # df = pd.read_sql("select user_id, product_id, rating from ai_labs_product_ratings", engine.connect())
    # ratings_matrix = df.pivot(index="user_id", columns="product_id", values="rating").fillna(0)

    df = pd.read_sql("""SELECT t1.id - 1 as user_id, t2.id - 1 as product_id, coalesce(t3.rating,0) as rating 
                        FROM ai_labs_user t1 
                        inner join ai_labs_product_info t2 
                        left join ai_labs_product_ratings t3 
                        on t1.id = t3.user_id and t2.id = t3.product_id
                        order by t1.id asc, t2.id asc""", engine.connect())

    num_users = df["user_id"].nunique()
    num_items = df["product_id"].nunique()

    # interactions = torch.tensor(df["rating"].values.reshape(num_users,num_items))
    # interactions = torch.FloatTensor(df.pivot(index="user_id", columns="product_id", values="rating").fillna(0).values)

    # user_ids = torch.LongTensor(df["user_id"].values)
    # item_ids = torch.LongTensor(df["product_id"].values)
    # ratings = torch.FloatTensor(df["rating"].values)
    
    model = CFModel(num_users, num_items, 10)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = CFDataset(df)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for epoch in range(20):
        for users, items, ratings in dataloader:
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


if __name__ == '__main__':

    with st.sidebar:
        tabs = st.tabs(["推荐设置"])
        with tabs[0]:
            mode = st.selectbox("推荐方式", key="config_recommend_mode", options=["协同过滤推荐", "基于内容推荐"])
            nums = st.slider("推荐个数", key="config_recommend_nums", min_value=1, max_value=10, value=5) 
    
    if mode == "协同过滤推荐":
        model = predict_model()
        items = recommend_items(model, st.session_state.userid, nums)
        st.write(f"为您推荐以下产品：")
        display_products(",".join([str(item + 1) for item in items]))
        
    if mode == "基于内容推荐":
        pass