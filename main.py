from interfaceDice import DiceCoin 
api='https://www.bitsler.com/api/bet-dice'
a= DiceCoin(api)
a.token='ab470734b6cea9cafcbb24fda2d3c65c68c9540eff7ec5927e2caad062e15d306cc58427fea227d0fe9c2e0a6858861ef55cde5d1917e944b31486cab7754e2b'
a.coin='btslr'
a.condition=50.49
a.guess=True
a.amount=1
a.crawlData()
