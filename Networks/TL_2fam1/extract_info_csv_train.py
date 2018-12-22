def get_info(path2file):
        to_return = dict()
        to_return["train_acc"] = []
        to_return["train_loss"] = []
        to_return["val_acc"] = []
        to_return["val_loss"] = []
        with open(path2file) as f:
                for line in f:
                        if "train" in line:
                                to_return["train_loss"].append(float(line.split()[2]))
                                to_return["train_acc"].append(float(line.split()[4]))
                        elif "val" in line and not "Best" in line: #avoid last line
                                to_return["val_loss"].append(float(line.split()[2]))
                                to_return["val_acc"].append(float(line.split()[4]))

        return(to_return)
d = get_info("TL_2fam1.out")

with open("../train_accuracy_TL2f1.csv", "a") as csv:
	csv.write("\nacc_TL2f1,")
	for i in range(len(d["train_acc"])):
		csv.write(str(d["train_acc"][i])+ ",")

with open("../train_accuracy_TL2f1.csv", "a") as csv:
	csv.write("\loss_TL2f1,")
	for i in range(len(d["train_loss"])):
		csv.write(str(d["train_loss"][i])+ ",")
        
with open("../accuracies.csv", "a") as csv:
	csv.write("\nacc_T2f1,")
	for i in range(len(d["val_acc"])):
		csv.write(str(d["val_acc"][i])+ ",")

with open("../accuracies.csv", "a") as csv:
	csv.write("\nloss_T2f1,")
	for i in range(len(d["val_loss"])):
		csv.write(str(d["val_loss"][i])+ ",")