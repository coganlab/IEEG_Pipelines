function specview(t,f,S,titl)
imagesc(t,f,S');
colormap(parula(4096));
%caxis([-2 2]);
set(gca,'YDir', 'normal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(titl);
