import difflib
file_1=open('/Users/wangyifan/Desktop/Bert_v2_2(upper_case).ipynb','r',encoding='utf-8').readlines()
file_2=open('/Users/wangyifan/Desktop/bert_v2_0(lowest_loss).ipynb','r',encoding='utf-8').readlines()
d=difflib.HtmlDiff()
results=d.make_file(file_1,file_2) # 返回HTML形式的比较字符串
with open('results.html','w') as file:
	file.write(results)	# 将比较结果保存在results.html文件中