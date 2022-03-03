import udkm.moke.functions as moke
import udkm.tools.functions as tools
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("C:/Users/Udkm/Documents/Code/udkm/udkm/tools/udkm_base.mplstyle")


def cartesian(r, theta):
    x = r*np.sin(np.radians(theta))
    z = r*np.cos(np.radians(theta))
    return x, z


# field parameter
r_field = 1
theta_array = np.arange(0, 90.1, 0.1)
length = len(theta_array)
# anis parameter
theta_anis = 90
r_anis = 0.2

# laser induced anis change
delta_r_anis = -r_anis/10


def model_torque_thermal(r_field, r_anis, delta_r_anis):
    theta_array = np.arange(0, 90.1, 0.1)
    length = len(theta_array)
    # anis parameter
    theta_anis = 90

    m = {}
    m["theta_field"] = theta_array
    m["r_field"] = r_field*np.ones(length)

    m["r_anis"] = r_anis*np.ones(length)
    m["theta_anis"] = theta_anis*np.ones(length)

    m["delta_r_anis"] = delta_r_anis*np.ones(length)

    m["x_anis_initial"], m["z_anis_initial"] = cartesian(m["r_anis"], m["theta_anis"])
    m["x_anis_final"], m["z_anis_final"] = cartesian((m["r_anis"]+m["delta_r_anis"]), m["theta_anis"])
    m["x_field"], m["z_field"] = cartesian(m["r_field"], m["theta_field"])

    m["x_initial"] = m["x_anis_initial"] + m["x_field"]
    m["y_initial"] = 0*np.ones(length)
    m["z_initial"] = m["z_anis_initial"] + m["z_field"]
    norm = np.sqrt(m["x_initial"]**2+m["y_initial"]**2+m["z_initial"])

    m["x_initial_norm"] = m["x_initial"]/norm
    m["y_initial_norm"] = m["y_initial"]/norm
    m["z_initial_norm"] = m["z_initial"]/norm

    m["x_final"] = m["x_anis_final"] + m["x_field"]
    m["y_final"] = 0*np.ones(length)
    m["z_final"] = m["z_anis_final"] + m["z_field"]

    m["torque"] = np.zeros((length, 3))
    m["torque_norm"] = np.zeros((length, 1))

    for i in range(length):
        magnetization = [m["x_initial_norm"][i], m["y_initial_norm"][i], m["z_initial_norm"][i]]
        field = [m["x_final"][i], m["y_final"][i], m["z_final"][i]]
        m["torque"][i, :] = np.cross(magnetization, field)
        m["torque_norm"][i, :] = np.linalg.norm(m["torque"][i, :])
    return m


# %% plot thermal mechanism torque
theta_anis = 90
r_anis = 0.2
r_field = 1

plt.figure()
for r_anis in [0.1, 1, 10, 100]:
    m = model_torque_thermal(r_field, r_anis, r_anis/10)
    plt.plot(m["theta_field"], m["torque_norm"], label=str(r_anis))
plt.legend(title=r"$\frac{\mathrm{H_{ani}}}{\mathrm{H_{ext}}}$", title_fontsize=14)
plt.xticks(np.arange(0, 91, 15))
plt.xlim([0, 90])
plt.ylim(bottom=0)
plt.xlabel(r"field angle $\phi\,\,(^\circ)$ ")
plt.ylabel(r"torque $|\vec{M}\times\vec{H}|$ (arb. units)")
plt.title("thermal mechanism")
plt.show()


# %%

def model_torque_strain(r_field, r_anis, delta_r_strain):
    theta_array = np.arange(0, 90.1, 0.1)
    length = len(theta_array)
    # anis parameter
    theta_anis = 90

    m = {}
    m["theta_field"] = theta_array
    m["r_field"] = r_field*np.ones(length)

    m["r_anis"] = r_anis*np.ones(length)
    m["theta_anis"] = theta_anis*np.ones(length)

    m["delta_r_anis"] = delta_r_anis*np.ones(length)

    m["x_anis_initial"], m["z_anis_initial"] = cartesian(m["r_anis"], m["theta_anis"])
    m["x_field"], m["z_field"] = cartesian(m["r_field"], m["theta_field"])
    m["x_anis_final"] = m["x_anis_initial"]
    m["z_anis_final"] = m["z_anis_initial"]

    m["x_strain"] = 0*np.ones(length)
    m["z_strain"] = -1*delta_r_strain*np.ones(length)

    m["x_initial"] = m["x_anis_initial"] + m["x_field"]
    m["y_initial"] = 0*np.ones(length)
    m["z_initial"] = m["z_anis_initial"] + m["z_field"]
    norm = np.sqrt(m["x_initial"]**2+m["y_initial"]**2+m["z_initial"])

    m["x_initial_norm"] = m["x_initial"]/norm
    m["y_initial_norm"] = m["y_initial"]/norm
    m["z_initial_norm"] = m["z_initial"]/norm

    m["x_final"] = m["x_anis_final"] + m["x_field"] + m["x_strain"]
    m["y_final"] = 0*np.ones(length)
    m["z_final"] = m["z_anis_final"] + m["z_field"] + m["z_strain"]

    m["torque"] = np.zeros((length, 3))
    m["torque_norm"] = np.zeros((length, 1))

    for i in range(length):
        magnetization = [m["x_initial_norm"][i], m["y_initial_norm"][i], m["z_initial_norm"][i]]
        #magnetization = [m["x_initial"][i], m["y_initial"][i], m["z_initial"][i]]
        field = [m["x_final"][i], m["y_final"][i], m["z_final"][i]]
        m["torque"][i, :] = np.cross(magnetization, field)
        m["torque_norm"][i, :] = np.linalg.norm(m["torque"][i, :])
    return m


# %% plot strain mechanism torque
theta_anis = 90
r_anis = 0.2
r_field = 1

plt.figure()
for r_anis in [0.1, 1, 10, 100]:
    m = model_torque_strain(r_field, r_anis, 0.1)
    #plt.plot(m["theta_field"], m["torque"][:,2], label=str(r_anis))
    plt.plot(m["theta_field"], m["torque_norm"], label=str(r_anis))
plt.legend(title=r"$\frac{\mathrm{H_{ani}}}{\mathrm{H_{ext}}}$", title_fontsize=14)
plt.xticks(np.arange(0, 91, 15))
plt.xlim([0, 90])
# plt.ylim(bottom=0)
plt.xlabel(r"field angle $\phi\,\,(^\circ)$ ")
plt.ylabel(r"torque $|\vec{M}\times\vec{H}|$ (arb. units)")
plt.title("strain mechanism")
plt.show()
