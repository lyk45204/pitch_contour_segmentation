function [InflecTime, InflecPitch] = FindAllInflectionPointFn(time,pitch,deriOrder)
%---------sample the pitch curve by finding the inflection points of the pitch trace where the second derivative (Force) change the sign ---------
x = time;
y = pitch;
dydx = gradient(y) ./ gradient(x);                                      % Derivative Of Unevenly-Sampled Data
dydx2 = gradient(dydx) ./ gradient(x);  
switch deriOrder
    case 1
        deri = dydx;
    case 2
        deri = dydx2;
end

zci = @(v) find(v(:).*circshift(v(:), [-1 0]) <= 0);                    % Returns Approximate Zero-Crossing Indices Of Argument Vector
zxidx = zci(deri);                                                      % Approximate Indices Where ‘dydx=0’
InflecTime = x(zxidx);
InflecPitch = y(zxidx);
% for k1 = 1:numel(zxidx)                                                 % Loop Finds ‘x’ & ‘y’ For ‘dydx=0’
%     ixrng = max(zxidx(k1)-2,1):min(zxidx(k1)+2,numel(x));
%     inflptx(k1) = interp1(dydx(ixrng), x(ixrng), 0, 'linear');
%     inflpty(k1) = interp1(x(ixrng), y(ixrng), inflptx(k1), 'linear');
% end
% 
% inflpts = sprintfc('(%5.3f, %5.3f)', [inflptx; inflpty].');
% text(inflptx, inflpty, inflpts,'FontSize',8, 'HorizontalAlignment','center', 'VerticalAlignment','bottom')