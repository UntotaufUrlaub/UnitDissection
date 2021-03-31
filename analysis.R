library(data.table)
library(ggplot2)

path_res_folder = '/home/jakob/Storage/uni/20WiSe/S-AML/Paper/Code/myres'
paths = list.files(path_res_folder, full.names = T)
names(paths) = basename(paths)
tables = lapply(paths, fread)
dt = rbindlist(tables[1:2]) # here just a subset of tables is selected. Adjust this to fit in your memory TODO: Adjust script to better handle big files.
colnames(dt) = c('activation', 'label')
# tablesCast = sapply(tables[1], function(t){dcast(t, V1~V2)})


labels = fread('/home/jakob/Storage/uni/20WiSe/S-AML/Paper/Code/NetDissect-Lite/dataset/broden1_224/label.csv')
dt[label != 0, name := labels[as.numeric(label)]$name]
dt[, label := as.factor(label)]

##########################
#### visualize the distributions with the highest median
##########################
# this should be the classes which have overall higher activations

dt[, median := median(activation), by = label]
medians = unique(dt$median)
amount = 20

show_distributions = function(dt){
  medians_to_keep = head(sort(medians, decreasing = T), amount)
  dt.s = dt[median %in% medians_to_keep] # remove all rows which have a label which has a very low mean
  dt.s[, name := as.factor(name)]
  dt.s$name = reorder(dt.s$name, dt.s$median)
  
  #ggplot(dt.s, aes(label, activation)) + geom_boxplot(outlier.shape = NA) + ylim(0, 0.2)
  g = ggplot(dt.s, aes(name, activation)) + geom_boxplot(outlier.shape = NA) + ylim(0, 0.15) + guides(x =  guide_axis(angle = 90))# + scale_y_log10()
  print(g)
  dt.s
}
dt.s = show_distributions(dt)

# The amount of observations for each of this levels is very different so only looking at the median might be misleading.
ggplot(dt.s[, .N, by='name'], aes(name, N)) + geom_bar(stat = 'identity') + guides(x =  guide_axis(angle = 90))
dt.s[, .N, by='name'][order(N)]

ggplot(dt.s[, .(N=.N/nrow(dt)), by='name'], aes(name, N)) + geom_bar(stat = 'identity') + guides(x =  guide_axis(angle = 90))
dt.s[, .(N=.N/nrow(dt)), by='name'][order(N)]

# remove the ones wich have to little of share of data. Then visualize the high means
dt[, data_share := .N/nrow(dt), by = label]
length(medians)
dt = dt[0.0001 < data_share] # only keep labels which have a reasonable data share
length(unique(dt$median)) # are there labels removed because of having to little share in the data?
show_distributions(dt)

##########################
#### distribution of labels per activation (bins)
##########################
# todo: maybe this analysis should been seperated for the groups (color, object etc.)

# assign the bin to each row
numberOfbins = 40 # what is a good number here? 20 is just an educated guess
# bins based on equal distance
brks <- seq(min(dt$activation), max(dt$activation), length=numberOfbins)
dt[,bin:=findInterval(activation, brks)]
# bins based on quantiles
quantiles <- quantile(dt$activation, seq(0,1,length.out = numberOfbins + 1))
dt[,bin:=findInterval(activation, quantiles)]

# dt # debugging
ggplot(dt, aes(as.factor(bin), activation)) + geom_boxplot(outlier.shape = NA)
# ggplot(dt[name %in% c('white-c', 'red-c')], aes(as.factor(bin), fill = as.factor(name))) + geom_bar(position='fill', show.legend = FALSE)
# ggplot(dt, aes(as.factor(bin), fill = as.factor(name))) + geom_bar(position='fill', show.legend = FALSE)

# calculate the share each label has in each distribution
dt[, binsize := .N, by='bin']
dt.2 = dt[, .N, by=c('bin', 'name', 'binsize')][, N2:=N/binsize]
dt.2 = dt.2[0.01 < N2] # 0.001 is the share each label would approximately have by randomness, as there are about 1000 classes. So to be of interest we should be above random

#show the shares for each bin.
ggplot(dt.2, aes(bin, N2, color = name)) + geom_line(show.legend = F) + labs(y = 'share', x = 'activation bin')# box plot and scatter plot don't seem useful here, so stick with the line plot
ggplot(dt.2[!is.na(name) & bin != 21], aes(bin, N2, color = name)) + geom_line(show.legend = T) # box plot and scatter plot don't seem useful here, so stick with the line plot
head(dt.2[bin==numberOfbins][order(N2, decreasing = T)][, .(name, N2)], 20) # using this you can hopefully see, which of the high activation lines corresponds to which label. Showing the label is hardly possible as there are more than 1000 classes
head(dt.2[bin==numberOfbins+1][order(N2, decreasing = T)][, .(name, N2)], 20) # using this you can hopefully see, which of the high activation lines corresponds to which label. Showing the label is hardly possible as there are more than 1000 classes

labels_with_highest_median = head(dt[, name,by=c('median','label', 'name')][order(median, decreasing = T)],5)$name
ggplot(dt.2[!is.na(name) & bin != 21], aes(bin, N2, color = name)) + geom_line(show.legend = FALSE) + facet_grid(~name %in% labels_with_highest_median) + scale_y_log10()

# plot how the change is between the bins. So which share rice or change dependent on activation. This seems like a correlation between activation and bin. Or the variance
dt.3 = dt.2[, .(c(0,diff(N2)), bin),by='name']
ggplot(dt.3[bin != 21 & !is.na(name)], aes(bin, V1, color = name)) + geom_line(show.legend = FALSE) + facet_grid(~name %in% labels_with_highest_median)
# the median and the rise don't really seem to agree. And i don't see a clear pattern in derivative to look for.
library(GGally)
dt.4 = dcast(dt.3[bin != 21 & !is.na(name)],...~name, value.var = 'V1')
res = sapply(dt.4[,-'bin'], function(v){
  cor(dt.4$bin, v, use="complete.obs", method = 'spearman')
  # cor(dt.4$bin, v, use="complete.obs")
})
res.2 = data.table(res = res, name = names(res))
res.5 = res.2[order(res, decreasing = T)][0.5 <= res]
# a lot of labels have very high correlation. This doesn't seem to provide a godd hint how to interprete the neuron.

# maybe the variance is good. Or both combined
res.3 = sapply(dt.4[,-'bin'], function(v){
  var(v, na.rm = T)
  # cor(dt.4$bin, v, use="complete.obs")
})
res.4 = data.table(var = res.3, name = names(res.3))
merge(res.4, res.5, all.y = T)[order(var, decreasing = T)]

# show how many datapoints are in each bin.
ggplot(dt.2, aes(bin, binsize)) + geom_line() + scale_y_log10() + annotation_logticks(sides = 'l')
min(dt$binsize)