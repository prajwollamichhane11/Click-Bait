import time
import datetime
from datetime import  fromtimestamp

timestamp = [1554742548133, 1554742538950, 1554742269461, 1554742256643, 1554742248905, 1554741992898, 1554741986025, 1554741983584, 1554741873162, 1554741862026]
print(type(timestamp))
date = datetime.fromtimestamp(timestamp[0])
print(date)
