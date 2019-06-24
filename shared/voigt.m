function [F,J] = voigt(x, xdata)
% Pseudo-Voigt Function
% V = nu * L(x,x0,w) + (1-nu)*G(x,x0,w)
% voigt([xo, w, nu], x)

% This software is intended for research and educational purposes. Any sale
% or commercial distribution is strictly forbidden, unless the Department
% of Mathematical Modelling, Tecnical University of Denmark has given
% explicit permission.
% 
% The software is provided "as-is". Support is generally not available.
% No warranties of any kind, express or implied, are made as to it or any 
% medium it may be on. No remedy will be provided for indirect, 
% consequential, punitve or incidental damages arising from it, including
% such from negligence, strict liability, or breach of warranty or
% contract, even after notice of the possibility of such damages.
%
% Copyright (C) 2014
% Tommy Sonne Alstrøm
% Technical University of Denmark (DTU)
%
% $Revision$ $DateTime$
nx = size(x,2);
if( nx == 4 )
    % first parm is weight
    a = x(:,1);
    x = x(:,2:4);
else
    a = ones(size(x,1),1);
end
x0 = x(:,1);
w  = x(:,2);
nu = x(:,3);
N = length(xdata);
NV = size(x0,1);
% expand matrices to add support for multiple voigts
xdata = repmat(xdata,NV,1);
x0 = repmat(x0,1,N);
w = repmat(w,1,N);
nu = repmat(nu,1,N);
a = repmat(a,1,length(xdata));

% calculate terms that are used multiple times
diff = xdata-x0;
kern = diff./w;
diff2 = kern.^2;

L = 1./(1 +diff2);
Gkern = -log(2).*diff2;
G = exp(Gkern);
nuL = nu.*L; % later caching
nuG = (1-nu).*G; % later caching
anu = a.*nu;
anu1 = a.*(1-nu);
F = anu.*L + anu1.*G;

if( nargout==2 )
    % reserve mem
    J = zeros(size(x,1)*nx,N);
    % inxJ controls where the derivatives are stored in J
    inxJ = 1:nx:nx*size(x,1);
    if( nx==4 )
        % amplitudes was specified, add derivatives for those as well
        inxJ = (inxJ)+1;
    end
    % calculate terms that are used multiple times
    w2 = w.^2;
    diffw2 = diff./w2;
    anu2L2 = 2.*anu.*L.^2;
    anu2GkernG = 2.*log(2).*anu1.*Gkern.*G;
    

    % derivative for centerpoint
    % A * nu * dL/dxo + A * (1-nu) * dG/dxo
    J(inxJ,:) = anu2L2.*diffw2 - anu2GkernG.*diffw2;
    inxJ = inxJ+1;
    
    % derivative for width
    % A * nu * dL/dw + A * (1-nu) * dG/dw
    J(inxJ,:) = anu2L2.*diff2./w - anu2GkernG.*diff2./w;
    inxJ = inxJ+1;

    % derivative for nu
    % A*L -A*G
    J(inxJ,:) = a.*L-a.*G;

    if( nx==4 )
        inxJ = 1:nx:nx*size(x,1);
        % derivative for amplitude
        % nu*L + (1-nu)*G
        J(inxJ,:) = nuL+nuG;
    end
end