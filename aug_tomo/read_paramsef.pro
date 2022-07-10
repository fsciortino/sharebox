PRO read_paramsef, filename, r, z, pfm, dnorm_dpar, fact, lines, A0, num_iterat


OPENR,lun_res,filename,/get_lun

leng_r = 0L
leng_z = 0L
version = ' '

READF,lun_res, leng_r, leng_z, version

r = FLTARR(leng_r+1)
FOR i=0,leng_r DO BEGIN
   r_bla = 0.
   READF,lun_res,r_bla
   r[i] = r_bla
ENDFOR

z = FLTARR(leng_z+1)
FOR i=0,leng_z DO BEGIN
   z_bla = 0.
   READF,lun_res,z_bla
   z[i] = z_bla
ENDFOR


pfm = FLTARR(leng_r+1,leng_z+1)
dnorm_dpar = FLTARR(leng_r+1,leng_z+1)
fact = FLTARR(leng_r+1,leng_z+1)
FOR i=0,leng_r DO $
 FOR j=0,leng_z DO BEGIN
    pfm_bla = 0.
    dnorm_dpar_bla = 0.
    fact_bla = 0.
    READF,lun_res, pfm_bla, dnorm_dpar_bla, fact_bla
    pfm[i,j] = pfm_bla
    dnorm_dpar[i,j] = dnorm_dpar_bla
    fact[i,j] = fact_bla
ENDFOR

trash = ' '
READF,lun_res,trash

n_lines = 0L
READF,lun_res,n_lines

rstart = FLTARR(n_lines)
zstart = FLTARR(n_lines)
phistart = FLTARR(n_lines)
rend = FLTARR(n_lines)
zend = FLTARR(n_lines)
phiend = FLTARR(n_lines)
fR_om = FLTARR(n_lines)
fZ_om = FLTARR(n_lines)
fPhi_om = FLTARR(n_lines)
iN = FLTARR(n_lines)
los_meas = FLTARR(n_lines)
rel_error = FLTARR(n_lines)
on = FLTARR(n_lines)
corr_fact = FLTARR(n_lines)

FOR i=0,n_lines-1 DO BEGIN
   rstart_bla = 0.
   zstart_bla = 0.
   phistart_bla = 0.
   rend_bla = 0.
   zend_bla = 0.
   phiend_bla = 0.
   fR_om_bla = 0.
   fZ_om_bla = 0.
   fPhi_om_bla = 0.
   iN_bla = 0.
   los_meas_bla = 0.
   rel_error_bla = 0.
   on_bla = 0.
   corr_fact_bla = 0.
   READF,lun_res, rstart_bla, zstart_bla, phistart_bla, rend_bla, zend_bla, phiend_bla, fR_om_bla, fZ_om_bla, fPhi_om_bla, iN_bla, los_meas_bla, rel_error_bla, on_bla, corr_fact_bla
   rstart[i] = rstart_bla
   zstart[i] = zstart_bla
   phistart[i] = phistart_bla
   rend[i] = rend_bla
   zend[i] = zend_bla
   phiend[i] = phiend_bla
   fR_om[i] = fR_om_bla
   fZ_om[i] = fZ_om_bla
   fPhi_om[i] = fPhi_om_bla
   iN[i] = iN_bla
   los_meas[i] = los_meas_bla
   rel_error[i] = rel_error_bla
   on[i] = on_bla
   corr_fact[i] = corr_fact_bla
ENDFOR


lines = {rstart:rstart, zstart:zstart, phistart:phistart, rend:rend, zend:zend, phiend:phiend, $
         los_meas:los_meas, rel_error:rel_error, on:on, corr_fact:corr_fact}

READF,lun_res,trash

A0=0.
READF,lun_res,A0

READF,lun_res,trash
READF,lun_res,trash

num_iterat=0L
READF,lun_res,num_iterat,trash
print,num_iterat
print,trash

CLOSE,lun_res
FREE_LUN,lun_res

END
