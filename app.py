from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle

app = Flask(__name__)



@app.route('/')
def home():
    return render_template("FormFields - front end.html")

@app.route("/predict",methods=['POST'])
def predict():
    
    model = pickle.load(open('model.pkl','rb'))
    data = pd.read_csv("list_data.csv")
    data = data.iloc[:,2:]
    data.fillna('b',inplace=True)
    values = dict((key, request.form.getlist(key)) for key in request.form.keys())
    
        
    
    testdf = pd.DataFrame(values)
    cols = list(data.columns)
    testdf = testdf.loc[:,cols]
    data = pd.concat([data,testdf])
    data = pd.get_dummies(data)
    test = data.iloc[-1].values.reshape(1, -1)
    testpred = model.predict(test)
    # pht =build_table(testdf.T,"blue_light")
    return render_template("predict.html",values=testdf.T.to_html(),pred = testpred[0])


if __name__=="__main__":
    app.run(debug=True,port = 6000)
  






