import pickle
path='./oj.pkl' 
	   
f=open(path,'rb')
data=pickle.load(f)
print(len(data))
for key in data.keys():
    print(key)