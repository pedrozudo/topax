def simp(rho, simp_penalty, rho_min):
    return (1 - rho_min) * (rho**simp_penalty) + rho_min
