import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np
import math
import matplotlib.pyplot as plt

class KalmanFilterRobot:
    def __init__(self):
        rospy.init_node('kalman_filter_robot', anonymous=True)
        
        # Parâmetros do robô
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Inicializa a posição atual e a covariância estimada
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.P_est_prev = np.eye(3)  # Covariância inicial
        self.dt = 0.1 # Intervalo de tempo (10 Hz)
        
        # Erros do sistema
        self.e_v = 0.00524  # erro na velocidade linear
        self.e_a = 0.00175  # erro na velocidade angular
        self.R = 0.01  # Ruído de saída
        self.Q = np.diag([0.01, 0.01, 0.01])  # Covariância do ruído de processo

        # Listas para armazenar posições reais e estimadas
        self.real_positions = []
        self.estimated_positions = []

    def odom_callback(self, msg):
        # Atualiza a posição atual do robô
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Converte orientação de quaternion para ângulo em radianos
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, self.theta) = euler_from_quaternion(orientation_list)

    def predicao(self, est_prev, u_prev, P_est_prev):
       
        xp_prev, yp_prev, theta_prev = est_prev
        v, w = u_prev
        dt = self.dt

        L = 0.360  # Distância entre as rodas do robô
        R_wheel = 0.05  # Raio das rodas (ajuste conforme necessário)

        # Calcula as velocidades reais das rodas
        W_r = (v + (w * L) / 2) / R_wheel
        W_l = (v - (w * L) / 2) / R_wheel

        # Atualiza o ângulo theta com base nas velocidades reais das rodas
        theta_prev += (W_r - W_l) * R_wheel * dt / L

        # Matriz A do modelo linearizado
        A = np.array([
            [1, 0, -v * np.sin(theta_prev) * dt],
            [0, 1,  v * np.cos(theta_prev) * dt],
            [0, 0,  1]
        ])
        B = np.array([
            [np.cos(theta_prev) * dt, 0],
            [np.sin(theta_prev) * dt, 0],
            [0, dt]
        ])
        W_prev = np.array([
            [self.e_v * np.cos(theta_prev) * dt],
            [self.e_v * np.sin(theta_prev) * dt],
            [self.e_a]
        ])
        
        # Estado predito
        x_pred_k = np.dot(A, est_prev) + np.dot(B, u_prev)
        P_pred_k = np.dot(A, np.dot(P_est_prev, A.T)) + self.Q

        return x_pred_k, P_pred_k

    def leitura_da_saida(self, x_k):
        xp_k, yp_k, theta_k = x_k
        dt = self.dt

        #C = np.eye(3)
        C = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])
        V_k = np.array([
            [self.e_v * np.cos(theta_k) * dt],
            [self.e_v * np.sin(theta_k) * dt],
            [self.e_a]
        ])
        
        y_k = np.dot(C, x_k) + V_k.flatten()
        #y_k = np.dot(C, x_k) + V_k.flatten()[:2].sum()
        return y_k

    def estimativa(self, x_pred_k, P_pred_k, y_k):
        C = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])
        R = np.eye(3) * self.R

        S = np.dot(C, np.dot(P_pred_k, C.T)) + R
        K_k = np.dot(P_pred_k,np.dot(C.T,np.linalg.inv(S)))
        #K_k = np.dot(P_pred_k, C.T) / (np.dot(C, np.dot(P_pred_k, C.T)) + self.R)

        # Estado estimado
        #x_est_k = x_pred_k + K_k * (y_k - np.dot(C, x_pred_k))
        x_est_k = x_pred_k + np.dot(K_k ,(y_k - np.dot(C, x_pred_k)))
        # Matriz identidade
        I = np.eye(3)
        #L_k = I - np.outer(K_k, C)
        L_k = I - np.dot(K_k,C)
        # Covariância do erro estimado
        P_est_k = np.dot(L_k, P_pred_k)
        
        return x_est_k, P_est_k

    def move_to_goal(self, x_goal, y_goal):
        rate = rospy.Rate(10)
        move_cmd = Twist()
        
        est_prev = np.array([self.x, self.y, self.theta])
        P_est_prev = self.P_est_prev

        while not rospy.is_shutdown():
            distance = math.sqrt((x_goal - self.x)**2 + (y_goal - self.y)**2)
            angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
            
            if distance > 0.1:
                v = min(0.5, distance)
                w = angle_to_goal - self.theta
                move_cmd.linear.x = v
                move_cmd.angular.z = w
                self.cmd_vel_pub.publish(move_cmd)
                
                x_pred_k, P_pred_k = self.predicao(est_prev, [v, w], P_est_prev)
                y_k = self.leitura_da_saida([self.x, self.y, self.theta])
                x_est_k, P_est_k = self.estimativa(x_pred_k, P_pred_k, y_k)

                # Atualiza estimativa e covariância para a próxima iteração
                est_prev = x_est_k
                P_est_prev = P_est_k

                # Salva posições reais e estimadas para o gráfico
                self.real_positions.append((self.x, self.y))
                self.estimated_positions.append((x_est_k[0], (x_est_k[1])))
                
                # Exibe a posição real e a posição estimada no terminal
                rospy.loginfo(f"Posição real: x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}")
                rospy.loginfo(f"Posição estimada: x={x_est_k[0]:.2f}, y={(x_est_k[1]):.2f}, theta={x_est_k[2]:.2f}")
            else:
                rospy.loginfo("Destino alcançado!")
                break

            rate.sleep()

        # Plota o gráfico ao final do movimento
        self.plot_positions()

    def plot_positions(self):
        # Extrai as coordenadas reais e estimadas para o gráfico
        real_x, real_y = zip(*self.real_positions)
        est_x, est_y = zip(*self.estimated_positions)
        
        plt.figure()
        plt.plot(real_x, real_y, label='Deslocamento Real', color='blue')
        plt.plot(est_x, est_y, label='Deslocamento Estimado', color='orange', linestyle='--')
        plt.xlabel('Posição x (m)')
        plt.ylabel('Posição y (m)')
        plt.title('Deslocamento Real vs Estimado')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    try:
        robot = KalmanFilterRobot()
        x_goal = float(input("Digite a coordenada x de destino: "))
        y_goal = float(input("Digite a coordenada y de destino: "))
        robot.move_to_goal(x_goal, y_goal)
    except rospy.ROSInterruptException:
        pass
