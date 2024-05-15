load("intrinsic_out.mat");
close all;
t = t';

figure();
hold on;
plot(t, q(:, 1));
xlabel("Time (s)");
ylabel("\bf{q}");
% yscale log;
hold off;