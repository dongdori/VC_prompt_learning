from subprocess import check_output

train = 'python main.py --device cuda --dataset eurosat --type {type} --kshot 16 --start_epoch 0 --division base --layer {layer}'
evaluate = 'python evaluate.py --device cpu --dataset eurosat --epoch 100 --type {type} --division {div} --kshot 16 --topk 1 --layer {layer}'
dynamic = 'text+vision_metanet'
static = 'text+vision'

for i in range(2,12):
	print("#############################################################")
	print(train.format(type = dynamic, layer = i))
	check_output(train.format(type = dynamic, layer = i), shell=True)
	print("*************************************************************")
	print(check_output(evaluate.format(type = dynamic, div = 'base', layer = i), shell=True).decode().splitlines()[-1])
	print(check_output(evaluate.format(type = dynamic, div = 'novel', layer = i), shell=True).decode().splitlines()[-1])
	print("#############################################################")
	print(train.format(type = static, layer = i))
	check_output(train.format(type = static, layer = i), shell=True)
	print("*************************************************************")
	print(check_output(evaluate.format(type = static, div = 'base', layer = i), shell=True).decode().splitlines()[-1])
	print(check_output(evaluate.format(type = static, div = 'novel', layer = i), shell=True).decode().splitlines()[-1])
	print("#############################################################")