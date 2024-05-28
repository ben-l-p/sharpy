case1 = load("case_data_test_swept_tip_65.mat");
case2 = load("case_data_test_swept_tip_77.mat");

close all;

t1 = case1.data.Intrinsic.t';
q1 = case1.data.Intrinsic.q;
ra1 = case1.data.Intrinsic.ra;
t2 = case2.data.Intrinsic.t';
q2 = case2.data.Intrinsic.q;
ra2 = case2.data.Intrinsic.ra;

figure();
hold on;
plot(t1, ra1(:, 3, 21));
plot(t2, ra2(:, 3, 21));
legend(["N=20", "N=30"]);
xlabel("Time (s)");
ylabel("z (m)");
hold off;

mode_n = 10;
figure();
hold on;
plot(t1, q1(:, mode_n));
plot(t2, q2(:, mode_n));
legend(["N=20", "N=30"]);
xlabel("Time (s)");
ylabel("\bf{q}");
hold off;
