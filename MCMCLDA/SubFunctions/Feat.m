function circle(x,y,r)
try
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit);
h.Color = [0 1 0]
h.LineWidth = 5
catch exception
end