PRO read_resultef, filename, shot, diag, time, r, z, emiss, los_meas, los_calc, rel_error

OPENR,lun_res,filename,/get_lun

diag = ' '
shot = 0L
time = 0.

READF,lun_res,diag,shot,time,FORMAT='(A3,I0,F0)'

error = 0L

READF,lun_res,error

leng_r = 0L
leng_z = 0L

READF,lun_res,leng_r,leng_z

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


emiss = FLTARR(leng_r+1,leng_z+1)
FOR i=0,leng_r DO $
 FOR j=0,leng_z DO BEGIN
    emiss_bla = 0.
    READF,lun_res,emiss_bla
    emiss[i,j] = emiss_bla
ENDFOR


n_lines = 0L
READF,lun_res,n_lines

los_meas = FLTARR(n_lines)
los_calc = FLTARR(n_lines)
rel_error = FLTARR(n_lines)
corr_fact = FLTARR(n_lines)

FOR i=0,n_lines-1 DO BEGIN
   los_meas_bla = 0.
   los_calc_bla = 0.
   rel_error_bla = 0.
   corr_fact_bla = 0.
   READF,lun_res, los_meas_bla, los_calc_bla, rel_error_bla, corr_fact_bla
   los_meas[i] = los_meas_bla
   los_calc[i] = los_calc_bla
   rel_error[i] = rel_error_bla
   corr_fact[i] = corr_fact_bla
ENDFOR


CLOSE,lun_res
FREE_LUN,lun_res

END
