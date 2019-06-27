data = read.table("authorlist.txt", sep = "\n")
length(unique(data$V1))
