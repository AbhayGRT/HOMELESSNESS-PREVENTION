@app.route("/predict", methods=["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({"homelessnes_predict":prediction.tolist()})