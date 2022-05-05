function norm_data = minmaxscaler(bla)
norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
end