#include <iostream>
#include <array>
#include <cmath>

class KalmanFilter {
public:
    KalmanFilter(const std::array<std::array<double, 3>, 3>& Q,
                 const double R,
                 double dt, double e_v, double e_a)
        : Q(Q), R(R), dt(dt), e_v(e_v), e_a(e_a) {
        // Inicializar estado estimado, covariância e erro
        x_est_prev = {0.0, 0.0, 0.0};
        P_est_prev = {{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        }};
    }

    void predict(const std::array<double, 3>& x_est_prev,
                 const std::array<double, 2>& u_prev,
                 const std::array<std::array<double, 3>, 3>& P_est_prev,
                 std::array<double, 3>& x_pred_k,
                 std::array<std::array<double, 3>, 3>& P_pred_k) {
        double xp_prev = x_est_prev[0];
        double yp_prev = x_est_prev[1];
        double theta_prev = x_est_prev[2];
        double v = u_prev[0];
        double w = u_prev[1];

        // Matriz de transição A
        std::array<std::array<double, 3>, 3> A = {{
            {1.0, 0.0, -v * std::sin(theta_prev) * dt},
            {0.0, 1.0,  v * std::cos(theta_prev) * dt},
            {0.0, 0.0, 1.0}
        }};

        // Matriz de controle B
        std::array<std::array<double, 2>, 3> B = {{
            {std::cos(theta_prev) * dt, 0.0},
            {std::sin(theta_prev) * dt, 0.0},
            {0.0, dt}
        }};

        // Vetor de ruído de processo W_prev
        std::array<double, 3> W_prev = {
            e_v * std::cos(theta_prev) * dt,
            e_v * std::sin(theta_prev) * dt,
            e_a
        };

        // Predição do estado (x_pred_k)
        for (int i = 0; i < 3; i++) {
            x_pred_k[i] = A[i][0] * x_est_prev[0] + A[i][1] * x_est_prev[1] + A[i][2] * x_est_prev[2] +
                          B[i][0] * u_prev[0] + B[i][1] * u_prev[1];
        }

        // Predição da covariância (P_pred_k = A * P_est_prev * A^T + Q)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                P_pred_k[i][j] = Q[i][j];
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        P_pred_k[i][j] += A[i][k] * P_est_prev[k][l] * A[j][l];
                    }
                }
            }
        }
    }

    double get_measurement(const std::array<double, 3>& x_k) {
        double xp_k = x_k[0];
        double yp_k = x_k[1];
        double theta_k = x_k[2];

        // Vetor de medição C
       std::array<double, 3> C = {1.0, 1.0, 0.0};

        // Vetor de ruído de medição V_k
        double V_k = e_v * std::cos(theta_k) * dt + e_v * std::sin(theta_k) * dt + e_a;

        // Saída medida (y_k) com ruído de medição
        return C[0] * xp_k + C[1] * yp_k + C[2] * theta_k + V_k;
    }

    void update(const std::array<double, 3>& x_pred_k,
                const std::array<std::array<double, 3>, 3>& P_pred_k,
                double y_k,
                std::array<double, 3>& x_est_k,
                std::array<std::array<double, 3>, 3>& P_est_k) {
        // Vetor de medição C
        std::array<double, 3> C = {1.0, 1.0, 0.0};

        // Ganho de Kalman (K_k)
        std::array<double, 3> K_k = {0.0, 0.0, 0.0};
        double denominator = R;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                denominator += C[i] * P_pred_k[i][j] * C[j];
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                K_k[i] += P_pred_k[i][j] * C[j];
            }
            K_k[i] /= denominator;
        }

        // Estado estimado (x_est_k)
        for (int i = 0; i < 3; i++) {
            x_est_k[i] = x_pred_k[i] + K_k[i] * (y_k - (C[0] * x_pred_k[0] + C[1] * x_pred_k[1] + C[2] * x_pred_k[2]));
        }

        // Atualização da covariância (P_est_k)
        std::array<std::array<double, 3>, 3> I = {{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        }};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                P_est_k[i][j] = P_pred_k[i][j];
                for (int k = 0; k < 3; k++) {
                    P_est_k[i][j] -= K_k[i] * C[k] * P_pred_k[k][j];
                }
            }
        }
    }

private:
    std::array<std::array<double, 3>, 3> Q;  // Covariância do ruído de processo
    double R;  // Covariância do ruído de medição
    double dt;  // Intervalo de tempo
    double e_v;  // Erro na velocidade linear
    double e_a;  // Erro na velocidade angular

    std::array<double, 3> x_est_prev;  // Estimativa de estado anterior
    std::array<std::array<double, 3>, 3> P_est_prev;  // Covariância do estado anterior
};

int main() {
    double dt = 0.1;  // Intervalo de tempo
    double e_v = 0.05;  // Erro na velocidade linear
    double e_a = 0.01;  // Erro na velocidade angular

    std::array<std::array<double, 3>, 3> Q = {{
        {0.01, 0.0, 0.0},
        {0.0, 0.01, 0.0},
        {0.0, 0.0, 0.01}
    }};
    double R = 0.05;

    // Inicializar o filtro de Kalman
    KalmanFilter kf(Q, R, dt, e_v, e_a);

    // Estado estimado anterior
    std::array<double, 3> x_est_prev = {2.0, 3.0, M_PI / 4};

    // Entrada de controle
    std::array<double, 2> u_prev = {1.0, 0.1};

    // Covariância estimada anterior
    std::array<std::array<double, 3>, 3> P_est_prev = {{
        {0.1, 0.0, 0.0},
        {0.0, 0.1, 0.0},
        {0.0, 0.0, 0.1}
    }};

    // Predição do estado, erro e covariância no tempo k
    std::array<double, 3> x_pred_k;
    std::array<std::array<double, 3>, 3> P_pred_k;

    kf.predict(x_est_prev, u_prev, P_est_prev, x_pred_k, P_pred_k);

    // Obter medição simulada
    double y_k = kf.get_measurement(x_pred_k);

    // Atualizar estado e covariância estimados
    std::array<double, 3> x_est_k;
    std::array<std::array<double, 3>, 3> P_est_k;

    kf.update(x_pred_k, P_pred_k, y_k, x_est_k, P_est_k);

    // Imprimir resultados
    std::cout << "Estado estimado: ";
    for (double val : x_est_k) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
