res = read.table("TL_4fam1/fail.csv", header = T, sep = ",")
?table()
res$T_class = as.factor(res$T_class)
res$P_class = as.factor(res$P_class)

table(res$T_class, res$P_class, row.names = c("pred_cla"))

res$T_class
res$P_class
res$path

sort(res$path[grep("ocel", res$path)])
