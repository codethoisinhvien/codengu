from Service.Dice import DiceCoin 
api='https://www.bitsler.com/api/bet-dice'
a= DiceCoin(api)
a.token='7dcba85fb75418e64f107f89c1eb63197eb9811d1ac657ae65cf735d629a916ffd309248486935cb5ef36b5b0eee4c240d2e6cc19accda2b44d18c0e60344088'
a.coin='xrp'
a.condition=50.49
a.guess=True
a.amount=0.00015000
for i in range(10000):
      a.crawlData()
      print('times',i)

