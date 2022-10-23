%% 
clear all;
contain_zero = 0; % If have mz_eta0 for eta=0, set this para. as 1. Else set as 0.
fileId=fopen('mz.txt','r');
mz=fscanf(fileId,'%f ');
fclose(fileId);
if (contain_zero)
  fileId0=fopen('mz_eta0.txt','r');
  mz0=fscanf(fileId0,'%f ');
  fclose(fileId0);
end

%------------------------------------------------------------------------
% modify these paramters according to the simulation
eta_start = 0.1;
eta_end = 90;
eta_step = 100;
J_start = (8e5);
J_end = (3e8);
J_step = 1000;
thermal_step = 1;

alph = 0.005; % damping constant
Ms = 1000; % emu/cc
Hk = 4000;
thickness = 1e-7; % unit: cm
DL=0.3;

%------------------------------------------------------------------------

e_charge = 1.6e-19; % electron charge
hbar = 1.054e-27; % reduced plank constant
eta=[0:0.1:90]/180*pi;


%x = linspace(eta_start,eta_end,eta_step);
x = logspace(log10(eta_start),log10(eta_end),eta_step);
y = linspace(J_start,J_end,J_step);

[X,Y] = meshgrid(x,y);

mz  = reshape(mz, J_step*eta_step, thermal_step);
mz = mean(mz, 2);
mz = reshape(mz,J_step,eta_step);
contourf(X,Y,mz);
xlabel('\eta (бу)','FontSize',18);
ylabel('J_c (A/cm^2)','FontSize',18);
cb=colorbar;
cb.Label.String = 'M_z/M';
cb.Label.FontSize=18;
Jsw = zeros(eta_step,1);
for i=1:eta_step
    if isempty(find(mz(:,i)<-0.9))
        Jsw(i)=-1e5;
    else
        Jsw(i) = y(min(find(mz(:,i)<-0.9)));
    end
end

hold on

% analytical equation ref: https://doi.org/10.1038/s41598-020-58669-1
Jeta=alph*2*e_charge/hbar*Ms*thickness/DL*Hk./sin(eta); % spin polarization angle is eta (eq.11)
A=-1+sqrt(1+6.*alph^2.*cot(eta).^2);
B=sqrt(3+12./(1+sqrt(6*alph^2.*cot(eta).^2)));
Jeta_accurate=2.*A.*B.*Hk.*Ms.*e_charge.*thickness.*sec(eta).*tan(eta)./(9.*alph.*hbar.*DL); %(eq.8)
Jy=0.5*2*e_charge/hbar*Ms*thickness/DL*Hk; % spin is totally in plane
Jeta(1)=Jy;
Jeta_accurate(1)=Jy;
eq=plot(eta/pi*180,Jeta,'r','LineWidth',2);
legend(eq,'analytical equation')
xlim([0.1 90]);
ylim([0 2e8]);
% hold on
% plot(eta/pi*180,Jeta_accurate,'g','LineWidth',2);

figure(2)
plot(eta/pi*180,Jeta,'r','LineWidth',2);
hold on
if (contain_zero)
   x=[0,x];
   J0 = y(min(find(mz0<-0.9)));
   Jsw = [J0;Jsw];
end
scatter(x,Jsw,80,'s','k','LineWidth',2);
legend('analytical equation','simulation results')
xlabel('\eta (бу)','FontSize',18);
ylabel('J_{SW} (A/cm^2)','FontSize',18);

xlim([0.1 90]);
ylim([0 2e8]);

%% 

% %%------------------------------------------------------------------
% %%uncomment these lines to draw trajectory
% fileId=fopen('m_x.txt','r');
% mx=fscanf(fileId,'%f ');
% fclose(fileId);
% fileId=fopen('m_y.txt','r');
% my=fscanf(fileId,'%f ');
% fclose(fileId);
% fileId=fopen('m_z.txt','r');
% mz=fscanf(fileId,'%f ');
% fclose(fileId);
% SP = figure;
% [x, y, z]=sphere;
% mesh(x,y,z,'EdgeColor',[0.9,0.9,0.9]);
% alpha(0.3)  % set transparency
% axis equal
% set(gca,'linewidth',1.5)
% set(gca,'FontSize',12)
% hold on
% plot3(mx,my,mz,'LineWidth',1.5)
% xlabel('M_{x}/M','FontSize',16);
% ylabel('M_{y}/M','FontSize',16);
% zlabel('M_{z}/M','FontSize',16);
% campos([9,-11,4.8])  %'CameraPosition'
% %%------------------------------------------------------------------