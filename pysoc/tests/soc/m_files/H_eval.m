% Determination of H matrix for c=Hy, explicit extended null space method
% and exact local method based on Alstad et al. (2009) 
% Evaporator

% Index for CVs candidates same as Kariwala's (2008)
% 1 - P2   - Separator Operating Pressure [kPa]
% 2 - T2   - Product Temperature          [°C]
% 3 - T3   - Vapor Temperature            [°C]
% 4 - F2   - Product Flowrate             [Kg/min]
% 5 - F100 - Steam Flowrate               [Kg/min]
% 6 - T201 - Outlet Temperature of C.W    [°C]
% 7 - F3   - Circulating Flowrate         [Kg/min]
% 8 - F5   - Condensate Flowrate          [Kg/min]
% 9 - F200 - C.W Flowrate                 [Kg/min]
% 10 - F1  - Feed Flowrate                [Kg/min]
clear;close;clc
% load matrices_self_opt.mat                                                      % (1k)     Load Matrices Gy,Gyd,Juu,Jud
load ../mat_files/input/matrices_self_opt_100pts.mat                                                 % (100pts) Load Matrices Gy,Gyd,Juu,Jud
% Kariwala Matrices (For checking purposes)
% Juu  = [0.006 -.133;-.133 16.737];
% Jud  = [0.023 eps -0.001 ;-158.373 -1.161 1.484];
% Gy   = [-0.093 11.678; -0.052 6.559; -0.047 5.921; eps 0.141; -0.001 1.115; -0.094 2.170;...
%     -0.032 6.594; eps 0.859; 1 eps; eps 1];
% Gyd  = [-3.626 eps 1.972; -2.036 eps 1.108; -1.838 eps 1; 0.267 eps eps; -0.317 -0.018 0.020;...
%     -0.674 eps 1;-2.253 -0.066 0.673; -0.267 eps eps; eps eps eps; eps eps eps];
nyt  = size(Gy,1);                                                                % Total Number of measruments
nu   = size(Gy,2);                                                                % Remaining DOF
nd   = size(d,2);                                                                 % N° of Disturbances
F    = -((Gy/Juu)*Jud - Gyd);                                                     % Eq (15) Alstad et al. (2009): Sensitivity Matrix
FT   = [F*Wd Wny];                                                                % Eq (27) Alstad et al. Definition (F~)
Gt   = [Gy Gyd];                                                                  % G tilde , Augmented Plant G.
me   = [1.285 1 1 0.027 0.189 1 0.494 0.163 4.355 0.189];                         % Measurement Errors.

%--------------Linear combination of all measurements----------------%
% Alstad et al. Section 3
% Exact Local Method:
H = ((FT*FT')\Gy)/((Gy'/(FT*FT'))*Gy)*sqrtm(Juu);                                 % Eq (31) Alstad et al. Exact local method;
H = H/norm(H); H = H';                                                            % Scaling to have ||H|| = 1.
% Alstad et al. Section 4
% Extended Nullspace Method:
Mn_null    = eye(size(Gy,2));                                                     % Mn = I
Jt         = [sqrtm(Juu)   (sqrtm(Juu)/Juu)*Jud];                                 % Eq (34) J~;
H_null     = Mn_null\Jt*pinv(Wny\Gt)/Wny; H_null = H_null/norm(H_null);           % Eq (41) Extended Nullspace Method
% H_null     = Jt*pinv(Wny\Gt)/Wny; H_null = H_null/norm(H_null);                 % Eq (41) Extended Nullspace Method
% Loss Eval., Exact Local Method:
G_exact    = H*Gy;                                                                % Gain
Mn_exact   = sqrtm(Juu)/G_exact;                                                  % Exact Mn
Md_exact   = -(sqrtm(Juu)/G_exact)*H*F*Wd;                                        % Eq (20) Alstad et al.
Mny_exact  = -(sqrtm(Juu)/G_exact)*H*Wny;                                         % Eq (21) Alstad et al.
M_exact    = [Md_exact Mny_exact];                                                % Definition, Alstad et al. (Eq) (22);
Loss_exact = (0.5*(max(svd(M_exact)).^2));                                        % Worst-Case Loss, eq (23)
Avg_Loss_exact   = (1/(6*(nyt+nd)))*((norm(M_exact,'fro'))^2);
% Loss Eval., Extended Nullspace:
G_null     =   H_null*Gy;                                                         % Gain
Md_null    = -(sqrtm(Juu)/G_null)*H_null*F*Wd;                                    % Eq (20) Alstad et al.
Mny_null   = -(sqrtm(Juu)/G_null)*H_null*Wny;                                     % Eq (21) Alstad et al.
M_null     =  [Md_null Mny_null];                                                 % Definition, Alstad et al. (Eq) (22);
Loss_null  =  (0.5*((max(svd(M_null))).^2));                                      % Worst-Case Loss, eq. (23)
Avg_Loss_null   = (1/(6*(nyt+nd)))*((norm(M_null,'fro'))^2);


% Loss_exact
% Loss_null
% H
% H_null
% Avg_Loss_exact
% Avg_Loss_null
% %-------------Subsets of possible measurements--------------%

% All possible subsets of size 2 using Exact Local Method

index_CV2 = nchoosek(1:10,2);                                                      % Index for CV combinations
index_Wny2= nchoosek(me,2);                                                        % Index for Measrument errors combinations
H_ss2    = zeros(10,2); G_exact_ss2 = zeros(10,1); Mn_exact_ss2 = zeros(10,1);     % Preallocation
Md_exact_ss2 = zeros(10,3); Mny_exact_ss2 = zeros(10,2); M_exact_ss2 = zeros(10,5);% Preallocation
Loss_exact_ss2 = zeros(10,1); Avg_Loss_ss2 = zeros(10,1);
warn_condss2 = zeros(size(index_CV2,1),size(index_CV2,2)+1);
for u = 1:size(index_CV2,1)

Gy_ss2     =    Gy(index_CV2(u,:),:); Gyd_ss2 = Gyd(index_CV2(u,:),:);             % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_ss2    =    diag(index_Wny2(u,:));                                             % Wny for each subset combination     (changes every loop iter.)

F_ss2      =   -((Gy_ss2/Juu)*Jud - Gyd_ss2);                                      % Eq (15) Alstad et al. (2009)
FT_ss2     =    [F_ss2*Wd Wny_ss2];                                                % Eq (27) Alstad et al. Definition (F~)
                                             

H_ss2  = ((FT_ss2*FT_ss2')\Gy_ss2)/...
((Gy_ss2'/(FT_ss2*FT_ss2'))*Gy_ss2)*sqrtm(Juu);                                    % Eq (31) Alstad et al. Exact local method;
H_ss2  = H_ss2/norm(H_ss2); H_ss2 = H_ss2';                                        % Scaling to have ||H|| = 1.

if u == 43
    a = 1;
end


G_exact_ss2    = H_ss2*Gy_ss2;                                                     % Gain
Mn_exact_ss2   = sqrtm(Juu)/G_exact_ss2;                                           % Exact Mn
Md_exact_ss2   = -(sqrtm(Juu)/G_exact_ss2)*H_ss2*F_ss2*Wd;                         % Eq (20) Alstad et al.
Mny_exact_ss2  = -(sqrtm(Juu)/G_exact_ss2)*H_ss2*Wny_ss2;                          % Eq (21) Alstad et al.
M_exact_ss2    = [Md_exact_ss2 Mny_exact_ss2];                                     % Definition, Alstad et al. (Eq) (22);
Loss_exact_ss2(u) = 0.5*((max(svd(M_exact_ss2))).^2);                              % Worst-Case Loss, eq (23)
Avg_Loss_ss2(u)   = (1/(6*(2+nd)))*((norm(M_exact_ss2,'fro')).^2);                 % Average Loss, Kariwala et al. (2008) eq. (11)
end

Losses_ss2 = sortrows([Loss_exact_ss2 Avg_Loss_ss2 index_CV2]);


% All possible subsets of size 3 using Exact Local Method

index_CV3 = nchoosek(1:10,3);                                                      % Index for CV combinations
index_Wny3= nchoosek(me,3);                                                        % Index for Measrument errors combinations

Loss_exact_ss3 = zeros(1,size(index_CV3,1));                                       % Preallocation
Avg_Loss_ss3   = zeros(1,size(index_CV3,1));                                       % Preallocation
for u = 1:size(index_CV3,1)

Gy_ss3     =    Gy(index_CV3(u,:),:); Gyd_ss3 = Gyd(index_CV3(u,:),:);             % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_ss3    =    diag(index_Wny3(u,:));                                             % Wny for each subset combination     (changes every loop iter.)

F_ss3      =   -((Gy_ss3/Juu)*Jud - Gyd_ss3);                                      % Eq (15) Alstad et al. (2009)
FT_ss3     =    [F_ss3*Wd Wny_ss3];                                                % Eq (27) Alstad et al. Definition (F~)
                                             

H_ss3  = ((FT_ss3*FT_ss3')\Gy_ss3)/...
((Gy_ss3'/(FT_ss3*FT_ss3'))*Gy_ss3)*sqrtm(Juu);                                    % Eq (31) Alstad et al. Exact local method;
H_ss3  = H_ss3/norm(H_ss3); H_ss3 = H_ss3';                                        % Scaling to have ||H|| = 1.



G_exact_ss3    = H_ss3*Gy_ss3;                                                     % Gain
Mn_exact_ss3   = sqrtm(Juu)/G_exact_ss3;                                           % Exact Mn
Md_exact_ss3   = -(sqrtm(Juu)/G_exact_ss3)*H_ss3*F_ss3*Wd;                         % Eq (20) Alstad et al.
Mny_exact_ss3  = -(sqrtm(Juu)/G_exact_ss3)*H_ss3*Wny_ss3;                          % Eq (21) Alstad et al.
M_exact_ss3    = [Md_exact_ss3 Mny_exact_ss3];                                     % Definition, Alstad et al. (Eq) (22);
Loss_exact_ss3(u) = 0.5*((max(svd(M_exact_ss3))).^2);                              % Worst-Case Loss, eq (23)
Avg_Loss_ss3  (u)  = (1/(6*(3+nd)))*((norm(M_exact_ss3,'fro'))^2);                 % Average Loss, Kariwala et al. (2008) eq. (11)
end
Losses_ss3 = sortrows([Loss_exact_ss3' Avg_Loss_ss3' index_CV3]);



% All possible subsets of size 4 using Exact Local Method

index_CV4 = nchoosek(1:10,4);                                                      % Index for CV combinations
index_Wny4= nchoosek(me,4);                                                        % Index for Measrument errors combinations
Loss_exact_ss4 = zeros(1,size(index_CV4,1));                                       % Preallocation
Avg_Loss_ss4   = zeros(1,size(index_CV4,1));                                       % Preallocation

for u = 1:size(index_CV4,1)
u          =    187; 
Gy_ss4     =    Gy(index_CV4(u,:),:); Gyd_ss4 = Gyd(index_CV4(u,:),:);             % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_ss4    =    diag(index_Wny4(u,:));                                             % Wny for each subset combination     (changes every loop iter.)

F_ss4      =   -((Gy_ss4/Juu)*Jud - Gyd_ss4);                                      % Eq (15) Alstad et al. (2009)
FT_ss4     =    [F_ss4*Wd Wny_ss4];                                                % Eq (27) Alstad et al. Definition (F~)
                                             

H_ss4  = ((FT_ss4*FT_ss4')\Gy_ss4)/...
((Gy_ss4'/(FT_ss4*FT_ss4'))*Gy_ss4)*sqrtm(Juu);                                    % Eq (31) Alstad et al. Exact local method;
H_ss4  = H_ss4/norm(H_ss4); H_ss4 = H_ss4';                                        % Scaling to have ||H|| = 1.


G_exact_ss4       = H_ss4*Gy_ss4;                                                  % Gain
Mn_exact_ss4      = sqrtm(Juu)/G_exact_ss4;                                        % Exact Mn
Md_exact_ss4      = -(sqrtm(Juu)/G_exact_ss4)*H_ss4*F_ss4*Wd;                      % Eq (20) Alstad et al.
Mny_exact_ss4     = -(sqrtm(Juu)/G_exact_ss4)*H_ss4*Wny_ss4;                       % Eq (21) Alstad et al.
M_exact_ss4       = [Md_exact_ss4 Mny_exact_ss4];                                  % Definition, Alstad et al. (Eq) (22);
Loss_exact_ss4(u) = 0.5*((max(svd(M_exact_ss4))).^2);                              % Worst-Case Loss, eq (23)
Avg_Loss_ss4  (u) = (1/(6*(4+nd)))*((norm(M_exact_ss4,'fro'))^2);                  % Average Loss, Kariwala et al. (2008) eq. (11)
end
Losses_ss4 = sortrows([Loss_exact_ss4' Avg_Loss_ss4' index_CV4]);


% All possible subsets of size 5 using Exact Local Method

index_CV5 = nchoosek(1:10,5);                                                      % Index for CV combinations
index_Wny5= nchoosek(me,5);                                                        % Index for Measrument errors combinations
Loss_exact_ss5 = zeros(1,size(index_CV5,1));                                       % Preallocation
Avg_Loss_ss5   = zeros(1,size(index_CV5,1));                                       % Preallocation

for u = 1:size(index_CV5,1)

Gy_ss5     =    Gy(index_CV5(u,:),:); Gyd_ss5 = Gyd(index_CV5(u,:),:);             % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_ss5    =    diag(index_Wny5(u,:));                                             % Wny for each subset combination     (changes every loop iter.)

F_ss5      =   -((Gy_ss5/Juu)*Jud - Gyd_ss5);                                      % Eq (15) Alstad et al. (2009)
FT_ss5     =    [F_ss5*Wd Wny_ss5];                                                % Eq (27) Alstad et al. Definition (F~)
                                             

H_ss5  = ((FT_ss5*FT_ss5')\Gy_ss5)/...
((Gy_ss5'/(FT_ss5*FT_ss5'))*Gy_ss5)*sqrtm(Juu);                                    % Eq (31) Alstad et al. Exact local method;
H_ss5  = H_ss5/norm(H_ss5); H_ss5 = H_ss5';                                        % Scaling to have ||H|| = 1.


G_exact_ss5       = H_ss5*Gy_ss5;                                                  % Gain
Mn_exact_ss5      = sqrtm(Juu)/G_exact_ss5;                                        % Exact Mn
Md_exact_ss5      = -(sqrtm(Juu)/G_exact_ss5)*H_ss5*F_ss5*Wd;                      % Eq (20) Alstad et al.
Mny_exact_ss5     = -(sqrtm(Juu)/G_exact_ss5)*H_ss5*Wny_ss5;                       % Eq (21) Alstad et al.
M_exact_ss5       = [Md_exact_ss5 Mny_exact_ss5];                                  % Definition, Alstad et al. (Eq) (22);
Loss_exact_ss5(u) = 0.5*((max(svd(M_exact_ss5))).^2);                              % Worst-Case Loss, eq (23)
Avg_Loss_ss5  (u) = (1/(6*(5+nd)))*((norm(M_exact_ss5,'fro'))^2);                  % Average Loss, Kariwala et al. (2008) eq. (11)
end
Losses_ss5 = sortrows([Loss_exact_ss5' Avg_Loss_ss5' index_CV5]);



% All possible subsets of size 6 using Exact Local Method

index_CV6 = nchoosek(1:10,6);                                                      % Index for CV combinations
index_Wny6= nchoosek(me,6);                                                        % Index for Measrument errors combinations
Loss_exact_ss6 = zeros(1,size(index_CV6,1));                                       % Preallocation
Avg_Loss_ss6   = zeros(1,size(index_CV6,1));                                       % Preallocation
for u = 1:size(index_CV6,1)

Gy_ss6     =    Gy(index_CV6(u,:),:); Gyd_ss6 = Gyd(index_CV6(u,:),:);             % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_ss6    =    diag(index_Wny6(u,:));                                             % Wny for each subset combination     (changes every loop iter.)

F_ss6      =   -((Gy_ss6/Juu)*Jud - Gyd_ss6);                                      % Eq (15) Alstad et al. (2009)
FT_ss6     =    [F_ss6*Wd Wny_ss6];                                                % Eq (27) Alstad et al. Definition (F~)
                                             

H_ss6  = ((FT_ss6*FT_ss6')\Gy_ss6)/...
((Gy_ss6'/(FT_ss6*FT_ss6'))*Gy_ss6)*sqrtm(Juu);                                    % Eq (31) Alstad et al. Exact local method;
H_ss6  = H_ss6/norm(H_ss6); H_ss6 = H_ss6';                                        % Scaling to have ||H|| = 1.


G_exact_ss6       = H_ss6*Gy_ss6;                                                  % Gain
Mn_exact_ss6      = sqrtm(Juu)/G_exact_ss6;                                        % Exact Mn
Md_exact_ss6      = -(sqrtm(Juu)/G_exact_ss6)*H_ss6*F_ss6*Wd;                      % Eq (20) Alstad et al.
Mny_exact_ss6     = -(sqrtm(Juu)/G_exact_ss6)*H_ss6*Wny_ss6;                       % Eq (21) Alstad et al.
M_exact_ss6       = [Md_exact_ss6 Mny_exact_ss6];                                  % Definition, Alstad et al. (Eq) (22);
Loss_exact_ss6(u) = 0.5*((max(svd(M_exact_ss6))).^2);                              % Worst-Case Loss, eq (23)
Avg_Loss_ss6  (u) = (1/(6*(6+nd)))*((norm(M_exact_ss6,'fro'))^2);                  % Average Loss, Kariwala et al. (2008) eq. (11)
end
Losses_ss6 = sortrows([Loss_exact_ss6' Avg_Loss_ss6' index_CV6]);


% All possible subsets of size 7 using Exact Local Method

index_CV7 = nchoosek(1:10,7);                                                      % Index for CV combinations
index_Wny7= nchoosek(me,7);                                                        % Index for Measrument errors combinations
Loss_exact_ss7 = zeros(1,size(index_CV7,1));                                       % Preallocation
Avg_Loss_ss7   = zeros(1,size(index_CV7,1));                                       % Preallocation

for u = 1:size(index_CV7,1)

Gy_ss7     =    Gy(index_CV7(u,:),:); Gyd_ss7 = Gyd(index_CV7(u,:),:);             % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_ss7    =    diag(index_Wny7(u,:));                                             % Wny for each subset combination     (changes every loop iter.)

F_ss7      =   -((Gy_ss7/Juu)*Jud - Gyd_ss7);                                      % Eq (15) Alstad et al. (2009)
FT_ss7     =    [F_ss7*Wd Wny_ss7];                                                % Eq (27) Alstad et al. Definition (F~)
                                             

H_ss7  = ((FT_ss7*FT_ss7')\Gy_ss7)/...
((Gy_ss7'/(FT_ss7*FT_ss7'))*Gy_ss7)*sqrtm(Juu);                                    % Eq (31) Alstad et al. Exact local method;
H_ss7  = H_ss7/norm(H_ss7); H_ss7 = H_ss7';                                        % Scaling to have ||H|| = 1.


G_exact_ss7       = H_ss7*Gy_ss7;                                                  % Gain
Mn_exact_ss7      = sqrtm(Juu)/G_exact_ss7;                                        % Exact Mn
Md_exact_ss7      = -(sqrtm(Juu)/G_exact_ss7)*H_ss7*F_ss7*Wd;                      % Eq (20) Alstad et al.
Mny_exact_ss7     = -(sqrtm(Juu)/G_exact_ss7)*H_ss7*Wny_ss7;                       % Eq (21) Alstad et al.
M_exact_ss7       = [Md_exact_ss7 Mny_exact_ss7];                                  % Definition, Alstad et al. (Eq) (22);
Loss_exact_ss7(u) = 0.5*((max(svd(M_exact_ss7))).^2);                              % Worst-Case Loss, eq (23)
Avg_Loss_ss7  (u) = (1/(6*(7+nd)))*((norm(M_exact_ss7,'fro'))^2);                  % Average Loss, Kariwala et al. (2008) eq. (11)
end
Losses_ss7 = sortrows([Loss_exact_ss7' Avg_Loss_ss7' index_CV7]);


% All possible subsets of size 8 using Exact Local Method

index_CV8 = nchoosek(1:10,8);                                                      % Index for CV combinations
index_Wny8= nchoosek(me,8);                                                        % Index for Measrument errors combinations
Loss_exact_ss8 = zeros(1,size(index_CV8,1));                                       % Preallocation
Avg_Loss_ss8   = zeros(1,size(index_CV8,1));                                       % Preallocation

for u = 1:size(index_CV8,1)

Gy_ss8     =    Gy(index_CV8(u,:),:); Gyd_ss8 = Gyd(index_CV8(u,:),:);             % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_ss8    =    diag(index_Wny8(u,:));                                             % Wny for each subset combination     (changes every loop iter.)

F_ss8      =   -((Gy_ss8/Juu)*Jud - Gyd_ss8);                                      % Eq (15) Alstad et al. (2009)
FT_ss8     =    [F_ss8*Wd Wny_ss8];                                                % Eq (27) Alstad et al. Definition (F~)
                                             

H_ss8  = ((FT_ss8*FT_ss8')\Gy_ss8)/...
((Gy_ss8'/(FT_ss8*FT_ss8'))*Gy_ss8)*sqrtm(Juu);                                    % Eq (31) Alstad et al. Exact local method;
H_ss8  = H_ss8/norm(H_ss8); H_ss8 = H_ss8';                                        % Scaling to have ||H|| = 1.


G_exact_ss8       = H_ss8*Gy_ss8;                                                  % Gain
Mn_exact_ss8      = sqrtm(Juu)/G_exact_ss8;                                        % Exact Mn
Md_exact_ss8      = -(sqrtm(Juu)/G_exact_ss8)*H_ss8*F_ss8*Wd;                      % Eq (20) Alstad et al.
Mny_exact_ss8     = -(sqrtm(Juu)/G_exact_ss8)*H_ss8*Wny_ss8;                       % Eq (21) Alstad et al.
M_exact_ss8       = [Md_exact_ss8 Mny_exact_ss8];                                  % Definition, Alstad et al. (Eq) (22);
Loss_exact_ss8(u) = 0.5*((max(svd(M_exact_ss8))).^2);                              % Worst-Case Loss, eq (23)
Avg_Loss_ss8  (u) = (1/(7*(8+nd)))*((norm(M_exact_ss8,'fro'))^2);                  % Average Loss, Kariwala et al. (2008) eq. (11)
end
Losses_ss8 = sortrows([Loss_exact_ss8' Avg_Loss_ss8' index_CV8]);


% All possible subsets of size 9 using Exact Local Method

index_CV9 = nchoosek(1:10,9);                                                      % Index for CV combinations
index_Wny9= nchoosek(me,9);                                                        % Index for Measrument errors combinations
Loss_exact_ss9 = zeros(1,size(index_CV9,1));                                       % Preallocation
Avg_Loss_ss9   = zeros(1,size(index_CV9,1));                                       % Preallocation

for u = 1:size(index_CV9,1)

Gy_ss9     =    Gy(index_CV9(u,:),:); Gyd_ss9 = Gyd(index_CV9(u,:),:);             % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_ss9    =    diag(index_Wny9(u,:));                                             % Wny for each subset combination     (changes every loop iter.)

F_ss9      =   -((Gy_ss9/Juu)*Jud - Gyd_ss9);                                      % Eq (15) Alstad et al. (2009)
FT_ss9     =    [F_ss9*Wd Wny_ss9];                                                % Eq (27) Alstad et al. Definition (F~)
                                             

H_ss9  = ((FT_ss9*FT_ss9')\Gy_ss9)/...
((Gy_ss9'/(FT_ss9*FT_ss9'))*Gy_ss9)*sqrtm(Juu);                                    % Eq (31) Alstad et al. Exact local method;
H_ss9  = H_ss9/norm(H_ss9); H_ss9 = H_ss9';                                        % Scaling to have ||H|| = 1.


G_exact_ss9       = H_ss9*Gy_ss9;                                                  % Gain
Mn_exact_ss9      = sqrtm(Juu)/G_exact_ss9;                                        % Exact Mn
Md_exact_ss9      = -(sqrtm(Juu)/G_exact_ss9)*H_ss9*F_ss9*Wd;                      % Eq (20) Alstad et al.
Mny_exact_ss9     = -(sqrtm(Juu)/G_exact_ss9)*H_ss9*Wny_ss9;                       % Eq (21) Alstad et al.
M_exact_ss9       = [Md_exact_ss9 Mny_exact_ss9];                                  % Definition, Alstad et al. (Eq) (22);
Loss_exact_ss9(u) = 0.5*((max(svd(M_exact_ss9))).^2);                              % Loss, eq (23)
Avg_Loss_ss9  (u) = (1/(7*(9+nd)))*((norm(M_exact_ss9,'fro'))^2);                  % Average Loss, Kariwala et al. (2008) eq. (11)
end
Losses_ss9 = sortrows([Loss_exact_ss9' Avg_Loss_ss9' index_CV9]);



% Extended Nullspace...

% "Just enough" measurements, Original Nullspace Method:

indexCV_null5 = nchoosek(1:10,5);                                                  % Index for CV combinations
indexWny_null5= nchoosek(me,5);                                                    % Index for Measrument errors combinations
Md_null5      = zeros(size(Gy,2),size(d,2));                                       % Md = 0 for ny >= nu + nd.
Loss_null5    = zeros(1,size(indexCV_null5,1));                                    % Preallocation
svM           = zeros(size(indexCV_null5,1),1);                                    % Preallocation
svG           = svM;                                                               % Preallocation
rcondG        = svM;                                                               % Preallocation
Avg_Loss_null5= svM';                                                              % Preallocation
f_normn5      = svM';                                                              % Preallocation
for u = 1:size(indexCV_null5,1)

Gy_null5   =    Gy(indexCV_null5(u,:),:); Gyd_null5 = Gyd(indexCV_null5(u,:),:);   % Gy, Gyd for each subset combination (changes every loop iter.)
Wny_null5  =    diag(indexWny_null5(u,:));                                         % Wny for each subset combination     (changes every loop iter.)
Gt_null5   =    [Gy_null5 Gyd_null5];
Mn_null5   =    eye(size(Gy_null5,2));
H_null5    = Mn_null5\Jt/Gt_null5; auxH = H_null5;
H_null5    = H_null5/norm(H_null5,'fro');                                          % Scaling to have ||H|| = 1.
F_null5    =  -((Gy_null5/Juu)*Jud - Gyd_null5);                                   % F (sensitivity Matrix)
Erf_md     = auxH*Gt_null5 - Jt;                                                   % Error fcn, eq. (38)
Mny_null5  = (-Jt/Gt_null5)*Wny_null5;                                             % Eq (43) Alstad et al.                                        
Loss_null5(u) = 0.5*(max(svd([Md_null5  Mny_null5]))).^2;                          % Worst-Case Loss, eq (23)
svM (u)        = max(svd(Mny_null5));svM=svM';
svG (u)        = min(svd(Gt_null5)); svG = svG';
rcondG(u)      = rcond(Gt_null5); rcondG=rcondG';
Avg_Loss_null5  (u) = (1/(6*(5+nd)))*((norm(Mny_null5,'fro'))^2);                  % Average Loss, Kariwala et al. (2008) eq. (11)
f_normn5(u)   = norm(Erf_md,'fro');                                                % Frobenius Norm of E, i.e ||E||f
end
Losses_je = sortrows([Loss_null5' Avg_Loss_null5' indexCV_null5]);
f_norms5  = sortrows([f_normn5' indexCV_null5]);
s_valuesM  = sortrows([svM rcondG Loss_null5' indexCV_null5]);
s_valuesG  = sortrows([svG rcondG Loss_null5' indexCV_null5],'descend');

% Branch and Bound (ny = nu + nd) - Just Enough measurements, maximizing
% minimum SV for Gy~:
[s,index_maxminG,nsv] = bnb(Gt,1);                                                  % BnB for Maximizing MSV
[Loss_bnb3wc,sset,ops,ctime,flag] = b3wc(Gy,Gyd,Wd,Wny,Juu,Jud,inf,10);             % BnB for Worst Case Loss
[msv,sset_msv]        = b3msv(Gt,1);                                                % BnB for Maximizing MSV (new Version By Yi Cao)

Gt_bnb                = Gt(index_maxminG,:); 
Md                    = zeros(size(Gy,2),size(d,2));                                % Zero disturbance Loss
Mn                    = eye(size(Gy,2));                                            % Mn = I (Remark 2 pg 6/11 Alstad et al. 2009)
Jt                    = [sqrtm(Juu)   (sqrtm(Juu)/Juu)*Jud];                        % J~ eq. (34)
H_bnb                 = (Mn\Jt)/Gt_bnb; auxH_bnb= H_bnb; H_bnb = H_bnb/norm(H_bnb); % H  eq. (43)
Erf_bnb               = auxH_bnb*Gt_bnb - Jt;                                       % Error fcn, eq.(38) must be close to 0...(Remark 1 pg 6/11 (143))
Wny_bnb               = diag(me(index_maxminG));                                    % Weightening for measurment error for BnB-chosen subset.
Mny_bnb               = (-Jt/Gt_bnb)*Wny_bnb;                                       % Eq.    (44)
M_bnb                 = [Md Mny_bnb];                                               % M = [Md Mny]
Loss_bnb              = 0.5*(max(svd(M_bnb)).^2);                                   % Eq. (23), Loss => Worst-Case Scenario
Avg_Loss_bnb          = (1/(6*(5+nd)))*((norm(M_bnb,'fro'))^2);                     % Average Loss   - Kariwala et al. 2008 eq.(11)

% Extra measurements (ny > nu + nd) , Extended Nullspace

% Subsets of size 6:

index_CV6_null      = nchoosek(1:10,6);                                             % Index for CV combinations
index_Wny6_null     = nchoosek(me,6);                                               % Index for Measrument errors combinations
Loss_null_ss6       = zeros(1,size(index_CV6_null,1));                              % Preallocation
Avg_Loss_null_ss6   = Loss_null_ss6;                                                % Preallocation
for u = 1:size(index_CV6_null,1)
Wny_null_ss6 =  diag(index_Wny6_null(u,:));                                         % Error measurement diagonal matrix for each combination
Gy_null_ss6  =  Gy(index_CV6_null(u,:),:);                                          % Gy for each combination
Gyd_null_ss6 =  Gyd(index_CV6_null(u,:),:);                                         % Gyd for each combination
Gt_null_ss6  =  [Gy_null_ss6 Gyd_null_ss6];                                         % Augmented Plant (Gy~)
Mn_null_ss6  =  eye(size(Gy_null_ss6,2));                                           % Mn = I (Remark 2 pg 6/11 (pdf) Alstad et al. 2009)
F_null_ss6   =  -((Gy_null_ss6/Juu)*Jud - Gyd_null_ss6);                            % F (sensitivity matrix) eq. (27)   
H_null_ss6   =  (Mn_null_ss6\Jt)*pinv(Wny_null_ss6\Gt_null_ss6)/Wny_null_ss6;       % Eq. (41), explicit expression for H in extended nullspace method
H_null_ss6   =  H_null_ss6/norm(H_null_ss6,'fro');
Mny_null_ss6 =  -sqrtm(Juu)/(H_null_ss6*Gy_null_ss6)*H_null_ss6*Wny_null_ss6;       % Eq. (21), Mny.
M_null_ss6   =  Mny_null_ss6;                                                       % M = [Md Mny], but Md = 0 for extended nullspace wheny ny>=nu+nd
Loss_null_ss6(u)= 0.5*(max(svd(M_null_ss6)).^2);                                    % Eq. (23), Loss => Worst-Case Scenario
Avg_Loss_null_ss6(u) = (1/(6*(6+nd)))*((norm(M_null_ss6,'fro'))^2);                 % Average Loss   - Kariwala et al. 2008
end 

Losses_null_ss6     = sortrows([Loss_null_ss6' Avg_Loss_null_ss6' index_CV6_null]);

% Subsets of size 7:

index_CV7_null      = nchoosek(1:10,7);                                              % Index for CV combinations
index_Wny7_null     = nchoosek(me,7);                                                % Index for Measrument errors combinations
Loss_null_ss7       = zeros(1,size(index_CV7_null,1));                               % Preallocation
Avg_Loss_null_ss7   = Loss_null_ss7;                                                 % Preallocation
for u = 1:size(index_CV7_null,1)
Wny_null_ss7 =  diag(index_Wny7_null(u,:));                                          % Error measurement diagonal matrix for each combination
Gy_null_ss7  =  Gy(index_CV7_null(u,:),:);                                           % Gy for each combination
Gyd_null_ss7 =  Gyd(index_CV7_null(u,:),:);                                          % Gyd for each combination
Gt_null_ss7  =  [Gy_null_ss7 Gyd_null_ss7];                                          % Augmented Plant (Gy~)
Mn_null_ss7  =  eye(size(Gy_null_ss7,2));                                            % Mn = I (Remark 2 pg 6/11 (pdf) Alstad et al. 2009)
F_null_ss7   =  -((Gy_null_ss7/Juu)*Jud - Gyd_null_ss7);                             % F (sensitivity matrix) eq. (27)
H_null_ss7   =  (Mn_null_ss7\Jt)*pinv(Wny_null_ss7\Gt_null_ss7)/Wny_null_ss7;        % Eq. (41), explicit expression for H in extended nullspace method
Mny_null_ss7 =  -sqrtm(Juu)/(H_null_ss7*Gy_null_ss7)*H_null_ss7*Wny_null_ss7;        % Eq. (21), Mny.
M_null_ss7   =  Mny_null_ss7;                                                        % M = [Md Mny], but Md = 0 for extended nullspace wheny ny>=nu+nd
Loss_null_ss7(u)= 0.5*(max(svd(M_null_ss7)).^2);                                     % Eq. (23), Loss => Worst-Case Scenario
Avg_Loss_null_ss7(u) = (1/(6*(7+nd)))*((norm(M_null_ss7,'fro'))^2);                  % Average Loss   - Kariwala et al. 2008
end 

Losses_null_ss7     = sortrows([Loss_null_ss7' index_CV7_null]);
Avg_Losses_null_ss7 = sortrows([Avg_Loss_null_ss7' index_CV7_null]);

% Subsets of size 8:

index_CV8_null = nchoosek(1:10,8);                                                   % Index for CV combinations
index_Wny8_null= nchoosek(me,8);                                                     % Index for Measrument errors combinations
Loss_null_ss8       = zeros(1,size(index_CV8_null,1));                               % Preallocation
Avg_Loss_null_ss8   = Loss_null_ss8;                                                 % Preallocation
for u = 1:size(index_CV8_null,1)
Wny_null_ss8 =  diag(index_Wny8_null(u,:));                                          % Error measurement diagonal matrix for each combination
Gy_null_ss8  =  Gy(index_CV8_null(u,:),:);                                           % Gy for each combination
Gyd_null_ss8 =  Gyd(index_CV8_null(u,:),:);                                          % Gyd for each combination
Gt_null_ss8  =  [Gy_null_ss8 Gyd_null_ss8];                                          % Augmented Plant (Gy~)
Mn_null_ss8  =  eye(size(Gy_null_ss8,2));                                            % Mn = I (Remark 2 pg 6/11 (pdf) Alstad et al. 2009)
F_null_ss8   =  -((Gy_null_ss8/Juu)*Jud - Gyd_null_ss8);                             % F (sensitivity matrix) eq. (27)
H_null_ss8   =  (Mn_null_ss8\Jt)*pinv(Wny_null_ss8\Gt_null_ss8)/Wny_null_ss8;        % Eq. (41), explicit expression for H in extended nullspace method
Mny_null_ss8 =  -sqrtm(Juu)/(H_null_ss8*Gy_null_ss8)*H_null_ss8*Wny_null_ss8;        % Eq. (21), Mny.
M_null_ss8   =  Mny_null_ss8;                                                        % M = [Md Mny], but Md = 0 for extended nullspace wheny ny>=nu+nd
Loss_null_ss8(u)= 0.5*(max(svd(M_null_ss8)).^2);                                     % Eq. (23), Loss => Worst-Case Scenario
Avg_Loss_null_ss8(u) = (1/(6*(8+nd)))*((norm(M_null_ss8,'fro'))^2);                  % Average Loss   - Kariwala et al. 2008
end 

Losses_null_ss8     = sortrows([Loss_null_ss8' Avg_Loss_null_ss8' index_CV8_null]);


% Subsets of size 9:

index_CV9_null = nchoosek(1:10,9);                                                   % Index for CV combinations
index_Wny9_null= nchoosek(me,9);                                                     % Index for Measrument errors combinations
Loss_null_ss9       = zeros(1,size(index_CV9_null,1));                               % Preallocation
Avg_Loss_null_ss9   = Loss_null_ss9;                                                 % Preallocation
for u = 1:size(index_CV9_null,1)
Wny_null_ss9 =  diag(index_Wny9_null(u,:));                                          % Error measurement diagonal matrix for each combination
Gy_null_ss9  =  Gy(index_CV9_null(u,:),:);                                           % Gy for each combination
Gyd_null_ss9 =  Gyd(index_CV9_null(u,:),:);                                          % Gyd for each combination
Gt_null_ss9  =  [Gy_null_ss9 Gyd_null_ss9];                                          % Augmented Plant (Gy~)
Mn_null_ss9  =  eye(size(Gy_null_ss9,2));                                            % Mn = I (Remark 2 pg 6/11 (pdf) Alstad et al. 2009)
F_null_ss9   =  -((Gy_null_ss9/Juu)*Jud - Gyd_null_ss9);                             % F (sensitivity matrix) eq. (27)
H_null_ss9   =  (Mn_null_ss9\Jt)*pinv(Wny_null_ss9\Gt_null_ss9)/Wny_null_ss9;        % Eq. (41), explicit expression for H in extended nullspace method
Mny_null_ss9 =  -sqrtm(Juu)/(H_null_ss9*Gy_null_ss9)*H_null_ss9*Wny_null_ss9;        % Eq. (21), Mny.
M_null_ss9   =  Mny_null_ss9;                                                        % M = [Md Mny], but Md = 0 for extended nullspace wheny ny>=nu+nd
Loss_null_ss9(u)= 0.5*(max(svd(M_null_ss9)).^2);                                     % Eq. (23), Loss => Worst-Case Scenario
Avg_Loss_null_ss9(u) = (1/(6*(9+nd)))*((norm(M_null_ss9,'fro'))^2);                  % Average Loss   - Kariwala et al. 2008
end 

Losses_null_ss9     = sortrows([Loss_null_ss9' Avg_Loss_null_ss9' index_CV9_null]);


% Subsets of size 10 (not necessary, done in line 30):

index_CV10_null = nchoosek(1:10,10);                                                  % Index for CV combinations
index_Wny10_null= nchoosek(me,10);                                                    % Index for Measrument errors combinations
Loss_null_ss10       = zeros(1,size(index_CV10_null,1));                              % Preallocation
Avg_Loss_null_ss10   = Loss_null_ss10;                                                % Preallocation
for u = 1:size(index_CV10_null,1) 
Wny_null_ss10 =  diag(index_Wny10_null(u,:));                                         % Error measurement diagonal matrix for each combination
Gy_null_ss10  =  Gy(index_CV10_null(u,:),:);                                          % Gy for each combination
Gyd_null_ss10 =  Gyd(index_CV10_null(u,:),:);                                         % Gyd for each combination
Gt_null_ss10  =  [Gy_null_ss10 Gyd_null_ss10];                                        % Augmented Plant (Gy~)
Mn_null_ss10  =  eye(size(Gy_null_ss10,2));                                           % Mn = I (Remark 2 pg 6/11 (pdf) Alstad et al. 2009)
F_null_ss10   =  -((Gy_null_ss10/Juu)*Jud - Gyd_null_ss10);                           % F (sensitivity matrix) eq. (27)
H_null_ss10   =  (Mn_null_ss10\Jt)*pinv(Wny_null_ss10\Gt_null_ss10)/Wny_null_ss10;    % Eq. (41), explicit expression for H in extended nullspace method
Mny_null_ss10 =  -sqrtm(Juu)/(H_null_ss10*Gy_null_ss10)*H_null_ss10*Wny_null_ss10;    % Eq. (21), Mny.
M_null_ss10   =  Mny_null_ss10;                                                       % M = [Md Mny], but Md = 0 for extended nullspace wheny ny>=nu+nd
Loss_null_ss10(u)= 0.5*(max(svd(M_null_ss10)).^2);                                    % Eq. (23), Loss => Worst-Case Scenario
Avg_Loss_null_ss10(u) = (1/(6*(10+nd)))*((norm(M_null_ss10,'fro'))^2);                % Average Loss   - Kariwala et al. 2008
end 

Losses_null_ss10     = sortrows([Loss_null_ss10' Avg_Loss_null_ss10' index_CV10_null]);


% Too few measurements (ny < nu + nd)

% Subsets of 2 elements:
index_fewCV2  = nchoosek(1:10,2);                                                      % Index for CV combinations
index_fewWny2 = nchoosek(me,2);                                                        % Index for Measrument errors combinations
Loss_f2       = zeros(1,size(index_fewCV2,1));                                         % Preallocation
f_norm2       = Loss_f2; Avg_Loss_f2=Loss_f2;                                          % Preallocation
for k = 1:size(index_fewCV2,1) 
Wny_f2    = diag(index_fewWny2(k,:));                                                  % Error measurement diagonal matrix for each combination                        
Gy_f2     = Gy(index_fewCV2(k,:),:);                                                   % Gy for each combination
Gyd_f2 = Gyd(index_fewCV2(k,:),:);                                                     % Gyd for each combination
Gt_f = [Gy_f2 Gyd_f2];                                                                 % Augmented Plant (Gy~)
F_f2      =  -((Gy_f2/Juu)*Jud - Gyd_f2);                                              % F (sensitivity matrix)
H_f2      = Jt*pinv(Gt_f); auxH_f2 = H_f2; H_f2 = H_f2/norm(H_f2);                     % Eq.(46), H when ny<nu+nd
Erf_md_f2 = auxH_f2*Gt_f - Jt;                                                         % Error Function, Eq. (38)
Mny_f2    = -(sqrtm(Juu)/(H_f2*Gy_f2))*H_f2*Wny_f2;                                    % Mny, eq.(21)
Md_f2     = -(sqrtm(Juu)/(H_f2*Gy_f2))*H_f2*F_f2*Wd;                                   % Md,  eq.(20), not zero when ny<nu+nd: Remark 1 pg. 143.
M_f2      = [Md_f2 Mny_f2];
% M_f2 = [Mny_f2];                                                            % M = [Md Mny]
Loss_f2(k)   = 0.5*((max(svd(M_f2))).^2);                                              % Worst-Case Loss, eq.(23)
Avg_Loss_f2(k)   = (1/(6*(2+nd)))*((norm(M_f2,'fro'))^2);                              % Average Loss - Kariwala et al. 2008
f_norm2(k)   = norm(Erf_md_f2,'fro');                                                  % Frobenius Norm of Error, must be close to 0.
end
Loss_f2= Loss_f2';
f_norm2= f_norm2';
% Subsets of 3 elements:
index_fewCV3  = nchoosek(1:10,3);                                                      % Index for CV combinations
index_fewWny3 = nchoosek(me,3);                                                        % Index for Measrument errors combinations
Loss_f3       = zeros(1,size(index_fewCV3,1));                                         % Preallocation
f_norm3       = Loss_f3; Avg_Loss_f3=Loss_f3;                                          % Preallocation
for k = 1:size(index_fewCV3,1)
Wny_f3  = diag(index_fewWny3(k,:));                                                    % Error measurement diagonal matrix for each combination 
Gy_f3   = Gy(index_fewCV3(k,:),:);                                                     % Gy for each combination
Gyd_f3 = Gyd(index_fewCV3(k,:),:);                                                     % Gyd for each combination
Gt_f = [Gy_f3 Gyd_f3];                                                                 % Augmented Plant (Gy~)
F_f3    =  -((Gy_f3/Juu)*Jud - Gyd_f3);                                                % F (sensitivity matrix)
H_f3    = Jt*pinv(Gt_f); auxH_f3 = H_f3; H_f3 = H_f3/norm(H_f3);                       % Eq.(46), H when ny<nu+nd
Erf_md_f3 = auxH_f3*Gt_f - Jt;                                                         % Error Function, Eq. (38)
Mny_f3  = -(sqrtm(Juu)/(H_f3*Gy_f3))*H_f3*Wny_f3;                                      % Mny, eq.(21)
Md_f3   = -(sqrtm(Juu)/(H_f3*Gy_f3))*H_f3*F_f3*Wd;                                     % Md,  eq.(20), not zero when ny<nu+nd: Remark 1 pg. 143.
M_f3    = [Md_f3 Mny_f3];                                                              % M = [Md Mny]
Loss_f3(k) = 0.5*((max(svd(M_f3))).^2);                                                % Worst-Case Loss, eq.(23)
Avg_Loss_f3(k)   = (1/(6*(3+nd)))*((norm(M_f3,'fro'))^2);                              % Average Loss - Kariwala et al. 2008
f_norm3(k) = norm(Erf_md_f3,'fro');                                                    % Frobenius Norm of Error, must be close to 0.
end
Loss_f3=Loss_f3';
f_norm3=f_norm3';

% Subsets of 4 elements:
index_fewCV4  = nchoosek(1:10,4);                                                      % Index for CV combinations
index_fewWny4 = nchoosek(me,4);                                                        % Index for Measrument errors combinations
Loss_f4       = zeros(1,size(index_fewCV4,1));                                         % Preallocation
f_norm4       = Loss_f4; Avg_Loss_f4=Loss_f4;                                          % Preallocation
for k = 1:size(index_fewCV4,1)
Wny_f4  = diag(index_fewWny4(k,:));                                                    % Error measurement diagonal matrix for each combination
Gy_f4   = Gy(index_fewCV4(k,:),:);                                                     % Gy for each combination
Gyd_f4 = Gyd(index_fewCV4(k,:),:);                                                     % Gyd for each combination
Gt_f = [Gy_f4 Gyd_f4];                                                                 % Augmented Plant (Gy~)
F_f4    =  -((Gy_f4/Juu)*Jud - Gyd_f4);                                                % F (sensitivity matrix)
H_f4    = Jt*pinv(Gt_f); auxH_f4 = H_f4; H_f4 = H_f4/norm(H_f4);                       % Eq.(46), H when ny<nu+nd
Erf_md_f4 = auxH_f4*Gt_f - Jt;                                                         % Error Function, Eq. (38)
Mny_f4  = -(sqrtm(Juu)/(H_f4*Gy_f4))*H_f4*Wny_f4;                                      % Mny, eq.(21)
Md_f4   = -(sqrtm(Juu)/(H_f4*Gy_f4))*H_f4*F_f4*Wd;                                     % Md,  eq.(20), not zero when ny<nu+nd: Remark 1 pg. 143.
M_f4    = [Md_f4 Mny_f4];                                                              % M = [Md Mny]
Loss_f4(k) = 0.5*((max(svd(M_f4))).^2);                                                % Worst-Case Loss, eq.(23)
Avg_Loss_f3(k)   = (1/(6*(4+nd)))*((norm(M_f4,'fro'))^2);                              % Average Loss - Kariwala et al. 2008
f_norm4(k) = norm(Erf_md_f4,'fro');                                                    % Frobenius Norm of Error, must be close to 0.
end
Loss_f4 = Loss_f4';
f_norm4 = f_norm4';

% Subset 3w Kariwala (F2,F100,F200)

% 4 - F2   - Product Flowrate             [Kg/min]
% 5 - F100 - Steam Flowrate               [Kg/min]
% 9 - F200 - C.W Flowrate                 [Kg/min]

% v1 = 4; v2=8; v3 = 9;
% Wny_3w  = diag([me(v1) me(v2) me(v3)]);
% Gy_3w   = Gy([v1 v2 v3],:); Gyd_3w = Gyd([v1 v2 v3],:);
% F_3w    = -((Gy_3w/Juu)*Jud - Gyd_3w);                                                  % Eq (15) Alstad et al. (2009): Sensitivity Matrix
% FT_3w   = [F_3w*Wd Wny_3w];                                                             % Eq (27) Alstad et al. Definition (F~)
% Gt_3w   = [Gy_3w Gyd_3w];                                                               % G tilde , Augmented Plant G.
% 
% H_3w = ((FT_3w*FT_3w')\Gy_3w)/((Gy_3w'/(FT_3w*FT_3w'))*Gy_3w)*sqrtm(Juu);               % Eq (31) Alstad et al. Exact local method;
% H_3w = H_3w/norm(H_3w); 
% H_3w = H_3w';                                                                           % Scaling to have ||H|| = 1.
% 
% 
% H_3w_n    = Jt*pinv(Gt_3w); H_3w_n = H_3w_n/norm(H_3w_n); 
% 
% G_exact_3w       = H_3w*Gy_3w;                                                  % Gain
% Mn_exact_3w      = sqrtm(Juu)/G_exact_3w;                                        % Exact Mn
% Md_exact_3w      = -(sqrtm(Juu)/G_exact_3w)*H_3w*F_3w*Wd;                      % Eq (20) Alstad et al.
% Mny_exact_3w     = -(sqrtm(Juu)/G_exact_3w)*H_3w*Wny_3w;                       % Eq (21) Alstad et al.
% M_exact_3w       = [Md_exact_3w Mny_exact_3w];                                  % Definition, Alstad et al. (Eq) (22);
% Loss_exact_3w = 0.5*((max(svd(M_exact_3w))).^2);                              % Worst-Case Loss, eq (23)
% Avg_Loss_3w   = (1/(6*(7+nd)))*((norm(M_exact_3w,'fro'))^2);                  % Average Loss, Kariwala et al. (2008) eq. (11)
% debug1=1;
