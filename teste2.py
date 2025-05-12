import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self):
        rospy.init_node('kalman_filter')

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_medida = 0.0
        self.w_medida = 0.0

        self.dt = 0.1
        self.a = 0.5
        self.alpha = 0.1

        self.Q = 0.01  # ruído de processo
        self.R = 0.05  # ruído de medição

        self.e_v = 0.05  # erro da velocidade linear para ruído de medição
        self.e_a = 0.01  # erro da velocidade angular para ruído de medição

        self.estado_inicial = np.array([0.0, 0.0, 0.0])
        self.covariancia_inicial = np.eye(3)

        self.x_est = []
        self.y_est = []
        self.x_real = []
        self.y_real = []

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.theta) = euler_from_quaternion(orientation_list)
        self.v_medida = msg.twist.twist.linear.x
        self.w_medida = msg.twist.twist.angular.z

    def predicao(self, estado, entrada, P_est_prev):
        dt = self.dt
        a = self.a
        alpha = self.alpha
        v, w = entrada

        x_pred = estado[0] + v * np.cos(estado[2]) * dt + 0.5 * a * np.cos(estado[2]) * dt**2
        y_pred = estado[1] + v * np.sin(estado[2]) * dt + 0.5 * a * np.sin(estado[2]) * dt**2
        theta_pred = estado[2] + w * dt + 0.5 * alpha * dt**2

        x_pred_k = np.array([x_pred, y_pred, theta_pred])

        A = np.array([
            [1, 0, -v * np.sin(estado[2]) * dt],
            [0, 1,  v * np.cos(estado[2]) * dt],
            [0, 0, 1]
        ])

        Q_prev = np.eye(3) * self.Q
        P_pred_k = A @ P_est_prev @ A.T + Q_prev

        return x_pred_k, P_pred_k

    def leitura_da_saida(self, x_k):
        dt = self.dt
        xp_k, yp_k, theta_k = x_k

        C = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])

        V_k = np.array([
            [self.e_v * np.cos(theta_k) * dt],
            [self.e_v * np.sin(theta_k) * dt]
        ])

        y_k = C @ x_k + V_k.flatten()
        return y_k

    def estimativa(self, x_pred_k, P_pred_k, y_k):
        C = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])

        R = np.eye(2) * self.R

        S = C @ P_pred_k @ C.T + R
        K_k = P_pred_k @ C.T @ np.linalg.inv(S)
        x_est_k = x_pred_k + K_k @ (y_k - C @ x_pred_k)

        I = np.eye(3)
        P_est_k = (I - K_k @ C) @ P_pred_k

        return x_est_k, P_est_k

    def move_to_goal(self, x_goal, y_goal):
        rate = rospy.Rate(1 / self.dt)

        est_prev = self.estado_inicial
        P_est_prev = self.covariancia_inicial

        while not rospy.is_shutdown():
            distancia = np.sqrt((x_goal - self.x)**2 + (y_goal - self.y)**2)
            if distancia < 0.1:
                break

            v = 0.5 * distancia
            w = 2 * (np.arctan2(y_goal - self.y, x_goal - self.x) - self.theta)

            v = np.clip(v, -0.5, 0.5)
            w = np.clip(w, -1.0, 1.0)

            vel_msg = Twist()
            vel_msg.linear.x = v
            vel_msg.angular.z = w
            self.cmd_vel_pub.publish(vel_msg)

            x_pred_k, P_pred_k = self.predicao(est_prev, [v, w], P_est_prev)
            y_k = self.leitura_da_saida([self.x, self.y, self.theta])
            x_est_k, P_est_k = self.estimativa(x_pred_k, P_pred_k, y_k)

            self.x_est.append(x_est_k[0])
            self.y_est.append(x_est_k[1])
            self.x_real.append(self.x)
            self.y_real.append(self.y)

            est_prev = x_est_k
            P_est_prev = P_est_k

            rate.sleep()

        vel_msg = Twist()
        self.cmd_vel_pub.publish(vel_msg)

        # Armazena dados
        self.x_est.append(x_est_k[0])
        self.y_est.append(x_est_k[1])
        self.x_real.append(self.x)
        self.y_real.append(self.y)

        # Cálculo do erro percentual
        erro_x = abs((self.x - x_est_k[0]) / self.x) * 100 if self.x != 0 else 0
        erro_y = abs((self.y - x_est_k[1]) / self.y) * 100 if self.y != 0 else 0
        erro_total = (erro_x + erro_y) / 2

        # Mostra no terminal
        rospy.loginfo(f"Erro percentual - X: {erro_x:.2f}%, Y: {erro_y:.2f}%, Médio: {erro_total:.2f}%")

        plt.figure()
        plt.plot(self.x_est, self.y_est, 'r-', label='Estimado (Kalman)')
        plt.plot(self.x_real, self.y_real, 'g--', label='Real (Odometria)')
        plt.plot(self.x_real[0], self.y_real[0], 'ko', label='Início')
        plt.plot(x_goal, y_goal, 'bx', label='Destino')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.title('Filtro de Kalman - Estimativa vs Realidade')
        plt.grid()
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    try:
        kf = KalmanFilter()
        rospy.sleep(1)
        kf.move_to_goal(-3.0, 5.0)
    except rospy.ROSInterruptException:
        pass
