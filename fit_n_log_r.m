%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Analysis of the cluster potential of Dehnen profiles S(\eta) and n(\beta)
% 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Units

G = 4.3009125e-3;     % pc * (km/s)^2 / Msun
R_norm = 1.226296320; % pc
V_norm = 4.587314029212518; % km/s

M_norm = 6.0e3; % Msun
omega = 231.38/8173; % km/s / pc
kappa = 1.37 * omega;
nu = 2.81 * omega;
gamma_sq = 4 * omega^2 - kappa^2;
nu_sq = nu^2;

%%% Grid 

% spherical
N_bin = 500;
r_max = 100;
rg  = [0 logspace(-2, log10(r_max), N_bin)].';
rm  = (rg(1:end-1) + rg(2:end))/2;
d_r = diff(rg);
dV  = d_r .* rm.^2 * 4*pi;

n_sph = 0;
N_LM  = (n_sph+1)*(n_sph+2)/2;

%%% Snapshot
path = cell(9,1);
for i1=1:3,
    for i2=1:3,        
        path{i2+(i1-1)*3} = num2str([i1 i2], '~/Data/WORK/DF_remnants/gamma=2.0/run-2.0-0.20-%d%d/');
    end
end

RHJm = []; RHJs = [];
NNf = [];

% for i_snap = 0:100:1000;

i_snap = 200;

Mr_ans = [];
Nf = [];

rh_rJ = [];
for ip=1:length(path),
    
    name = num2str(i_snap, [path{ip} '%.6d.dat']);
    
    %%% Read def-dc.dat
    
    DC   = load([path{ip} 'def-dc.dat']);
    r_dc = DC(:,3:5);
    rdc = r_dc(i_snap+1,:);    
    
    %%% Read stars
    
    fid = fopen(name,'rt');
    A_ = textscan(fid, '%f', 'HeaderLines',3);
    fclose(fid);
    A = reshape(A_{1},19,length(A_{1})/19).';    
    M   = A(:,2)*M_norm;
    r_v = A(:,3:5)*R_norm;
    
    %% Spherical harmonics - Mass distribution
    
    Rm_lm   = zeros(N_bin, N_LM);
    Rn_lm   = zeros(N_bin, N_LM);
    
    x_rel  = r_v(:,1)-rdc(1);
    y_rel  = r_v(:,2)-rdc(2);
    z_rel  = r_v(:,3)-rdc(3);
    r2_rel = x_rel.^2 + y_rel.^2 + z_rel.^2;
    r_rel = sqrt(r2_rel);
    
    [Ph, R_rel] = cart2pol(x_rel, y_rel);
    cos_Th = z_rel./r_rel;
    Th = acos(cos_Th);
    
    j=0;    
    for l = 0:n_sph,
        P = legendre(l, cos_Th);        
        for m = 0:l,
            Norm = sqrt( (2*l+1)/4/pi * exp( gammaln(l-m+1)-gammaln(l+m+1) ) );
            Ys_lm = Norm * P(m+1,:).' .* exp( 1i*m*Ph );
            
            j = j+1;
            for ir = 1:N_bin,
                II = find( r_rel>=rg(ir) & r_rel<rg(ir+1) );
                if ~isempty(II)
                    Rm_lm(ir,j) = sum(Ys_lm(II).*M(II));
                    Rn_lm(ir,j) = sum(Ys_lm(II));
                end
            end
        end
    end
    
        %% Multipole expansion (see Binney-Tremaine, Sect. 2.4, page 78)
    
    % R_lm = 3/4/pi./(1+rm.^2).^(2.5).*dV./sqrt(4*pi);
    
    P_lm = zeros(length(rg), l+1);
    j=0;
    for l=0:n_sph,
        
        % calculation of the potential
        
        for m=0:l,
            
            j=j+1;
            %        dMdR = csaps(rm, (-1)^m * Rho(:, j)/d_r, smooth_rho, rm);
            dMn = Rm_lm(:, j);
            f1 = rm.^l .* dMn;
            
            F10 = cumsum(f1);
            F11 = [0; F10];
%             F1  = (F11(2:end) + F11(1:end-1))/2./rg(2:end).^(l+1);
            F1  = F11./rg.^(l+1); F1(1)=0;
            
            f2 = dMn./rm.^(l+1);
            F20 = cumsum(f2);
            F21 = [0; F20];
            F22 = (F21(end)-F21);
   %         F2  = (F22(2:end) + F22(1:end-1))/2.*rg(2:end).^l;
            F2  = F22.*rg.^l;
            P_lm(:, j) = -4*pi*G*( F1 + F2 )/(2*l+1);            
            
        end
    end
        
    
%     %% pot check
%     eps = 1e-6;
%     rp = rg;
%     Phi_x = nan(size(rp));
%     Phi_y = nan(size(rp));
%     Phi_z = nan(size(rp));
%     
%     for ip = 1:length(rp),
%         
%         Phi_x(ip) = -G*sum(M ./sqrt(eps+ (x_rel-rp(ip)).^2 + y_rel.^2 + z_rel.^2));
%         Phi_y(ip) = -G*sum(M ./sqrt(eps+ (y_rel-rp(ip)).^2 + x_rel.^2 + z_rel.^2));
%         Phi_z(ip) = -G*sum(M ./sqrt(eps+ (z_rel-rp(ip)).^2 + y_rel.^2 + x_rel.^2));
%         
%     end
%     
%     Phi = (Phi_x + Phi_y+Phi_z)/3;
%     
%     loglog(rg, P_lm(:,1)./sqrt(4*pi), 'k', ...
%         rp, Phi_x, 'r', ...
%         rp, Phi_y, 'g', ...
%         rp, Phi_z, 'b')
% 

    % monopole density
    rho  = Rm_lm(:,1) ./dV./sqrt(4*pi);
    

    %%% Use further number or mass distribution    
    R_lm = Rn_lm;
    
        
    dM = R_lm(:,1)*sqrt(4*pi);
    Mr = [0; cumsum(dM)];
    Mr_ans = [Mr_ans Mr/length(r_rel)];
    
%     rdc = r_dc(i_snap+1,:);
%     plot(r_v(:,1), r_v(:,2), '.', rdc(1), rdc(2), 'xr')
%     Rb = 9000;
%     axis(Rb*[-1 1 -1 1])
    
    
    %% Jacobi radius
    
    Phi =  P_lm(:,1)./sqrt(4*pi);
    
    [tmp, I_max] = max(Phi - gamma_sq.*rg.^2/2);
    r_J = rm(I_max);
    
    IJ = find(rg<r_J);
    N_fraction_J = Mr_ans(IJ(end), end);
    r_h = csaps(Mr_ans(IJ, end), rg(IJ), 1, N_fraction_J/2);
    Nf = [Nf N_fraction_J];
    rh_rJ = [rh_rJ r_h/r_J];

end

RHJm = [RHJm mean(rh_rJ)];
RHJs = [RHJs std(rh_rJ)];
NNf = [NNf mean(Nf)];

% end

%% load Dehnen profiles

fid = fopen('../Data/dehnen_n.dat','rt');
D = textscan(fid, '%f', 'HeaderLines',1);
fclose(fid);
Dehnen = reshape(D{1},7,501).';
eta =  Dehnen(:,1);

n1 = Dehnen(:,2);
n2 = Dehnen(:,4);
n3 = Dehnen(:,6);

fid = fopen('../Data/dehnen_s.dat','rt');
D = textscan(fid, '%f', 'HeaderLines',1);
fclose(fid);
Dehnen = reshape(D{1},7,501).';
eta =  Dehnen(:,1);
dn1 = eta.*Dehnen(:,2);
dn2 = eta.*Dehnen(:,4);
dn3 = eta.*Dehnen(:,6);

%% plots

%%% fit gamma=0
% a_1 = 0.3227; a3rho0_1 = 1/n1(end)/a_1;
a_1 = 1.55; a3rho0_1 = 1.1/n1(end)/a_1;

%%% fit gamma=1
a_2 = 0.513; a3rho0_2 = 1/n2(end)/a_2;

%%% fit gamma=2
a_3 = 1.24; a3rho0_3 = 1/n3(end)/a_3;

%%% fit King

f = @(r, k, rc, rt) k * ( 1./sqrt(1+(r/rc).^2) - 1./sqrt(1+(rt/rc).^2) ).^2;

mean_Mr = mean(Mr_ans,2);
pp_Mr = csaps(rg, mean_Mr, 1-0.00001);
mean_Rh = diff(mean_Mr)./dV;
pp_Rh = csaps(rm, mean_Rh, 1-0.000001);

b = logspace(-2, 1.5, 101);
nb = b*0;
bbb = [b.'];
for ib = 1:length(b),
    
    z_max = sqrt(r_max^2 - b(ib)^2);
    ui = linspace(-10, log(z_max), 101);
    zi = exp(ui);
    ri = sqrt(zi.^2 + b(ib)^2);
    Szi = NewtonCotes(1,ui).*zi;
    nb(ib) = sum(Szi .* fnval(pp_Rh,ri))*4*pi*b(ib); 
    
end

%%% DDF

h = loglog(b, nb, 'r-', ...
    eta*a_1, a3rho0_1*dn1, 'k:', ...
    eta*a_2, a3rho0_2*dn2, 'm:', ...
    eta*a_3, a3rho0_3*dn3, 'g:', ...
    rm, 2*pi*rm.*f(rm, 0.55, 0.25, 1000),...
    eta*0.55, 7*S.*eta);
set(h(1), 'LineWidth', 3 )
set(h(2:5), 'LineWidth', 2 )

hold on;
ra = linspace(0.3e-2, 0.3, 21); plot(ra, 3.e-1.*ra.^1.0, 'k-.', 'LineWidth', 1)
ra = linspace(1,40,21);  plot(ra, 12.0e-1./ra.^2.0, 'k--', 'LineWidth', 1)
hold off;
axis([1e-4 max(rg) 5e-4 1e1]);

legend({'refined', '\gamma=0', '\gamma=1', '\gamma=2', 'King', '1/R', '1/R^2' },'FontSize',14,'Location','NorthEast');
legend('boxoff')


xlabel('R', 'FontSize', 14); 
ylabel('Diff. distr. dN(R)/dR', 'FontSize', 14); 

%%

clf
gamma1=0;
a=0.55;
loglog(rm, fnval(fnder(pp_Mr), rm), ...
    rm, 3.7*a^(-3) * rm.^2 .* (a./rm).^gamma1 .* (1+ rm/a).^(gamma1-4),...
    rm, 2* rm.^(-2) )

%%

% gamma1=0;
% rho_d = @(r) 1./r.^gamma1./(1+r).^(4-gamma1);
% 
% ur = linspace(-10,10,1001);
% rr = exp(ur);
% Sr = rr .* NewtonCotes(2,ur);
% 
% dM = 4*pi*rr.^2 .* rho_d(rr) .* Sr;
% Mc = [0 cumsum(dM)];
% rc = [0 rr];
% loglog(rc, Mc)
% 
% csaps(Mc, rc, 1, Mc(end)/2)
% csaps(rc, Mc/Mc(end), 1, R_norm/0.3227)