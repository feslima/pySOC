function [B,sset,ops,ctime,flag]=pb3wc(G1,Gd1,Wd,Wn,Juu,Jud,n,tlimit,nc)
% PB3WC Partial Bidirectional Branch and Bound (PB3) for Worst Case Criterion 
%
%   [B,sset,ops,ctime,flag] = pb3wc(G1,Gd1,Wd,Wn,Juu,Jud,n,tlimit,nc) is an
%   implementation of PB3 algorithm to select measurements, whose 
%   combinations can be used as controlled variables. The measurements are 
%   selected to provide minimum worst-case loss in terms of self-optimizing
%   control based on the following static optimization problem:
%
%       min J = f(y,u,d)
%       s.t.
%           y = G1 * u + G_d1 * Wd * d + Wn * e
%
%  where y is measurement vector (ny), u is input vector (nu), d is disturance
%  vector (nd) and e is the control error vector (ny).
%
%  The worst case loss is defined based on the assumption that 
%
%       ||d'  e'||_2 = 1 
%
% Inputs:
%   G1 = process model
%   Gd1 = disturbance model
%   Wd  = diagonal matrix containing magnitudes of disturbances
%   Wn = diagonal matrix containing magnitudes of implementation errors
%   Juu = \partial^2 J/\partial u^2
%   Jud = \partial^2 J/\partial u\partial d
%   n is the number of measurements to be selected, nu <= n <= ny.
%   tlimit defines the time limit to run the code. Default is Inf.
%   nc determines the number of best subsets to be selected. Default is 1.
%
% Outputs:
%   B     - nc x 1 vector of the wc loss of selected subsets.
%   sset  - nc x nu indices of selected subsets
%   ops   - number of nodes evaluated.
%   ctime - computation time used
%   flag  - 0 if successful, 1 otherwise (for tlimit < Inf).
%
% References:
%  [1]    I. J. Halvorsen, S. Skogestad, J. C. Morud, and V. Alstad.
%  Optimal selection of controlled variables. Ind. Eng. Chem. Res.,
%  42(14):3273-3284, 2003.  
%  [2]   V. Kariwala, Y. Cao, and S. Janardhanan. Local self-optimizing 
%  control with average loss minimization. Ind. Eng. Chem. Res., 
%  47(4):1150-1158, 2008.
%  [3]    V. Kariwala and Y. Cao, Bidirectional Branch and Bound for
%  Controlled Variable Selection: Part II. Exact Local Method for
%  Self-optimizing Control, Computers and Chemical Engineering, 
%  33(8):1402:1412, 2009 
%
%  See also b3wc, randcase, b3msv

%  By Yi Cao at Cranfield University, 8th January 2009
%

% Example.
%{
ny=40;
nu=15;
nd=5;
[G,Gd,Wd,Wn,Juu,Jud] = randcase(ny,nu,nd);
[B,sset,ops,ctime] = pb3wc(G,Gd,Wd,Wn,Juu,Jud,25);
% It takes about 0.9 seconds
%}


% Default inputs and outputs
    flag=false;
    if nargin<9
        nc=1;
    end
    if nargin<8
        tlimit=Inf;
    end
    ctime0=cputime;
    [r,m] = size(G1);
    if nargin<7
        n=m;
    end
    if n<m
        error('n must larger than number of inputs.')
    end
    % prepare matrices
    Y = [(G1*inv(Juu)*Jud-Gd1)*Wd Wn];
    G = G1*inv(sqrtm(Juu));
    Y2=Y*Y';
    Gd=G;
    Xd=Y2;
    Xu=Xd;
%     Z=Y2;
    h2=diag(Y2);
    q2=diag(G*G')./h2;
    p2=q2;
    ops=zeros(1,4);
    B=zeros(nc,1);
    sset=zeros(nc,n);
    ib=1;
    bound=0;
    % counters: 1) terminal; 2) nodes; 3) sub-nodes; 4) calls    
    ops=[0 0 0 0];
    fx=false(1,r);
    rem=true(1,r);
    downV=fx;
    downR=fx;
    nf=0;
    n2=n;
    m2=r;
    % initialize flags
    f=false;
    bf=false;
    downf=false;
    downff=false;
    upf=false;
    upff=true;
    
    [ny, nu] = size(G1);
    [~, nd] = size(Gd1);
    
    fileID = fopen('matlab.log', 'w+');
    fprintf(fileID, 'ny%d_nu%d_nd%d_n%d_nc%d\n\n',ny,nu,nd,n,nc);
    
    log_bbl3 = true;
    log_upprune = true;
    log_downprune = true;
    log_update = true;
    
    % the recursive solver
    bbL3sub(fx,rem);
    % convert bound to loss
    [B,idx]=sort(0.5./B);
    sset=sort(sset(idx,:),2);
    ctime=cputime-ctime0;
    
    fclose(fileID);
    save(['../mat_files/pb3wc/results/' sprintf('ny%d_nu%d_nd%d_n%d_nc%d.mat',ny,nu,nd,n,nc)], 'B','sset', 'ops', 'ctime', 'flag')
    
    function bn=bbL3sub(fx0,rem0)
        if log_bbl3
            fprintf(fileID, ['BBL3SUB\t\t- ', repmat('%g, ', 1, numel(ops)-1), '%g | %g | %g\n'], ops, sum(fx), sum(rem));
        end
        
        % recursive solver
        bn=0;
        if cputime-ctime0>tlimit
            flag=true;
            return;
        end
        ops(4)=ops(4)+1;
        fx=fx0;
        rem=rem0;
        nf=sum(fx);
        m2=sum(rem);
        n2=n-nf;
        while ~f && m2>n2 && n2>=0  % Loop for second branchs
            while ~f && m2>n2 && n2>0 && (~downf || ~upf || ~bf)
                % Loop for bidirection pruning
                if (~upf || ~bf) && n2 <= m && bound
                    % upwards pruning
                    upprune;
                else
                    upf=true;
                end
                if ~f &&  m2>n2 && m2>0 && (~downf || ~bf) && bound
                    % downwards pruning
                    downprune;
                else
                    downf=true;
                end
                bf=true;
            end %pruning loop
            if f || m2<n2 || n2<0
                % pruned cases
                return
            elseif m2==n2 || ~n2
                % terminal cases
                break
            end
            if n2==1,   % one more element to be fixed
                if upff
                    q2(~rem)=0;
                    [b,idk]=max(q2);
                else
                    p2(~rem)=Inf;
                    [b,idk]=min(p2);
                end
                s=fx;
                s(idk)=true;
                
                if log_bbl3
                    fprintf(fileID, ['BBL3SUB\t\t - s (fixed) - [', repmat('%g ', 1, numel(s)-1), '%g]\n'], s);
                end
                
                bn=update(s)-1;
                if bn>0
                    bn=bn+sum(fx0)-nf;
                    return
                end
                rem(idk)=false;
                m2=m2-1;
                downf=false;
                upf=true;
                continue
            end
            if m2-n2==1,        % one more element to be removed
                if downff
                    p2(~rem)=0;
                    [b,idk]=max(p2);
                else
                    q2(~rem)=Inf;
                    [b,idk]=min(q2);
                end
                rem(idk)=false;
                s=fx|rem;
%                 if sum(s)~=n
%                     disp(sum(s))
%                 end

                if log_bbl3
                    fprintf(fileID, ['BBL3SUB\t\t - s (removed) - [', repmat('%g ', 1, numel(s)-1), '%g]\n'], s);
                end
                
                update(s);
                fx(idk)=true;
                nf=nf+1;
                n2=n2-1;
                m2=m2-1;
                upf=false;
                downf=true;
                continue
            end
            % save data for bidirectional branching
            fx1=fx;
            rem1=rem;
%             b0=bound;
            p0=p2;
            q0=q2;
            D0=Xd;
            U0=Xu;
            dV0=downV;
            dR0=downR;            
            if n2-m<0.75*m2 %upward branching
                if bound
                    p2(~rem)=Inf;
                    [b,id]=min(p2);
                else
                    q2(~rem)=0;
                    [b,id]=max(q2);
                end
                fx(id)=true;
                rem1(id)=false;
                downf=true;
                upf=false;
                bn=bbL3sub(fx,rem1)-1;
                downf=false;
                upf=true;
            else        % downward branching
                if bound
                    p2(~rem)=0;
                    [b,id]=max(p2);
                else
                    q2(~rem)=Inf;
                    [b,id]=min(q2);
                end
                fx1(id)=true;
                downf=false;
                rem1(id)=false;
                downf=false;
                upf=true;
                bn=bbL3sub(fx,rem1)-1;
                downf=true;
                upf=false;
                if q0(id)<=bound && n==m
                    return
                end
            end
            % check pruning conditions
            if bn>0
                bn=bn-sum(rem0)+m2;
                return
            end
            % Recover data saved before the first branch
            fx=fx1;
            rem=rem1;
            f=false;
            p2=p0;
            q2=q0;
            nf=sum(fx);
            n2=n-nf;
            m2=sum(rem);
            Xd=D0;
            Xu=U0;
            downV=dV0;
            downR=dR0;
        end
        if ~f       % terminal cases
            if m2==n2
                bn=update(fx|rem)-1;
            elseif ~n2
                bn=update(fx)-1;
            end
        end
        bn=bn+sum(fx0)-nf;
    end

    function upprune
        if log_upprune
            fprintf(fileID, ['UPPRUNE\t\t- ', repmat('%g, ', 1, numel(ops)-1), '%g | %g | %g\n'], ops, sum(fx), sum(rem));
        end
        
        % Partially upwards pruning
        upf=true;
        [R1,f]=chol(Y2(fx,fx));     % nf x nf
        if f,
            return
        end               
        X1=R1'\G(fx,:);         % nf x m
        D=X1'*X1;               % m x m
        tD=trace(D);
        if tD<bound && n2<m     % m eigen values < bound
            ops(2)=ops(2)+1;
            f=true;
            return
        end
        if tD/m>bound && n2==m  % at least one eigen value > bound
            return
        end
        if m>2 %&& n2<m          % general cases
            bf0=sum(eig(D)<bound);
            ops(1)=ops(1)+1;
            if bf0>n2           % not feasible
                f=true;
                return
            end
            if bf0~=n2          % no pruning
                return
            end
            D=eye(m)*bound-D;
        else                    % special cases without using eig
            D=eye(m)*bound-D;
            [R,f]=chol(D);
            ops(2)=ops(2)+1;
            % ~f: m eigen values < bound
            % f: at lease 1 eigen value > bound
            if (f && n2>1) || ~f && n2<m
                f=~f;
                return;
            end
            if n2==1 % for m=2, nf=1
                [R,f]=chol(-D);
                if ~f
                    % m eigen values > bound, no pruning
                    return
                end
                % otherwise only one eigen value < bound
                f=~f;
            end
        end
        R2=R1'\Y2(fx,rem);
        R3=h2(rem)-sum(R2.*R2,1)';
        X2=G(rem,:)-R2'*X1;
        ops(3)=ops(3)+m2;
        q2(:)=Inf;
        if n2==1 || m>2
            q2(rem)=sum(X2'.*(D\X2'),1)'./R3-1;
        else
            X=R'\X2';
            q2(rem)=sum(X.*X,1)'./R3-1;
        end
        upff=true;
        L=q2<=0;
        if any(L)
            % upwards pruning
            downf=false;
            downff=false;
            rem(L)=false;
            m2=sum(rem);
            q2(L)=Inf;
        end
    end

    function downprune
        if log_downprune
            fprintf(fileID, ['DOWNPRUNE\t- ', repmat('%g, ', 1, numel(ops)-1), '%g | %g | %g\n'], ops, sum(fx), sum(rem));
        end
        
        % downwards pruning
        downf=true;
        s0=fx|rem;
        t=xor(downV,s0);
        if bf && sum(t)==1 && downR(t) %&& sum(downV)>nf+m2
            % single update
            D=Xd(rem,rem);
            x=Xd(rem,t);
            D=D-x*(x'/Xd(t,t));
            U=Xu(rem,rem);
            x=Xu(rem,t);
            U=U-x*(x'/Xu(t,t));
            downV=s0;
        elseif bf && isequal(downV,s0) && sum(downR&rem)==m2
            % no pruning
            return;
        else
            % normal cases
            [R1,f]=chol(Y2(s0,s0));
            if f,
                return
            end               
            Q=R1\(R1'\eye(m2+nf));
            downV=s0;
            Yinv(s0,s0)=Q;
            V=G(s0,:);
            U=Q*V;
            Gd(s0,:)=U;
            [R,f]=chol(V'*U-eye(m)*bound);
            if f,
                ops(2)=ops(2)+1;
                return
            end
            U=R'\Gd(rem,:)';
            D=Yinv(rem,rem);
            U=D-U'*U;            
        end
        ops(3)=ops(3)+m2;
        downR=rem;
        p2(rem)=diag(U)./diag(D);
        Xd(rem,rem)=D;
        Xu(rem,rem)=U;
        p2(~rem)=Inf;
        downff=true;
        L=p2<=0;
        if any(L)
            % downwards pruning
            upff=false;
            upf=false;
            fx(L)=true;
            rem(L)=false;
            nf=sum(fx);
            m2=sum(rem);
            n2=n-nf;
        end
    end

    function bf0=update(s)
        if log_update
            fprintf(fileID, ['UPDATE\t\t- ', repmat('%g, ', 1, numel(ops)-1), '%g | %g | %g | ib(%g)\n'], ops, sum(fx), sum(rem), ib);
        end
        
        % termal cases to update the bound
        X=chol(Y2(s,s))'\G(s,:);
        lambda=eig(X'*X);
        
%         lambda(abs(lambda) < eps) = 0;

        ops(1)=ops(1)+1;
        bf0=sum(lambda<bound);
        if ~bf0
            B(ib)=min(lambda);         %avoid sorting
            sset(ib,:)=find(s);
            bound0=bound;
            [bound,ib]=min(B);
            bf=bound0==bound;
        end
    end
end
