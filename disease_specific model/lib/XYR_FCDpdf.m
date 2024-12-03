function [binCounts,binEdges,fcdpdf] = XYR_FCDpdf(fcd)
%计算FCD的概率分布
fcd_mask = tril(ones(size(fcd,1),size(fcd,1)),0);
fx1 = fcd(~fcd_mask);
% x1  =  x1(~isnan(x1));
% x1  =  x1(:);
if isempty(fx1)
   error(message('stats:kstest2:NotEnoughData', 'X1'));
end
[binCounts,binEdges] = histcounts(fx1,'NumBins',10000,'BinLimits',[(0.0001-1),1]);
fcdpdf =  cumsum(binCounts);