FROM tensorflow/serving:2.10.1

COPY clothing-model /models/clothing-model/1 
ENV MODEL_NAME="clothing-model"