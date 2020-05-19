import tushare as ts

ts.set_token('5f9bdb72fd4dff979a64102a2f33f067c0118fc6fb4ce3001a27d5ae')
pro = ts.pro_api()
data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
print(data)
