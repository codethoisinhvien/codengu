import requests
class DiceCoin:
    def __init__(self,api):
        self.api= api
        self.token=2
        self.coin=2
        self.amount=3
        self.condition=4
        self.guess=True
    def coverObject(self):
        data= {'access_token':self.token,'currency':self.coin,'target':self.condition,'amount':self.amount,'over':self.guess}
        return data


    def crawlData(seft):
        res=requests.post(seft.api,seft.coverObject())
        print(res.status_code)
        if(res.status_code==200):
            val = res.json()
            print(val)
            print(val['username'])
        
        print()


