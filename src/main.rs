use peroxide::fuga::*;
use dialoguer::Select;

pub const MU: f64 = 398600.4418;    // Standard gravitational parameter of Earth
pub const R_EARTH: f64 = 6378.137;  // Radius of Earth in km
pub const J2: f64 = 1.08262668e-3;  // J2 coefficient of Earth

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let orbits = vec![
        OrbitType::LEO,
        OrbitType::GEO,
        OrbitType::Molniya,
    ];

    let selection = Select::new()
        .with_prompt("Select orbit type")
        .items(&orbits)
        .default(0)
        .interact()
        .unwrap();

    let selected_orbit = orbits[selection].create_orbit();
    let initial_state = selected_orbit.initial_state();

    let t0 = 0.0;
    let tf = 86400.0;
    let dt = 60.0;

    let problem = KeplerProblem;
    let yoshida_solver = YoshidaSolver::new(problem);
    let rk4 = RK4;
    let dp45 = DP45 {
        max_step_iter: 1000,
        max_step_size: 2.0 * dt,
        min_step_size: 1e-3 * dt,
        safety_factor: 0.9,
        tol: 1e-3,
    };

    let rk4_solver = BasicODESolver::new(rk4);
    let dp45_solver = BasicODESolver::new(dp45);

    let y0 = Vec::from(initial_state);
    y0.print();
    let (t_yoshida, y_yoshida) = yoshida_solver.solve(
        (t0, tf),
        dt,
        &y0,
    )?;
    let (t_rk4, y_rk4) = rk4_solver.solve(
        &problem,
        (t0, tf),
        dt,
        &y0,
    )?;
    let (t_dp45, y_dp45) = dp45_solver.solve(
        &problem,
        (t0, tf),
        dt,
        &y0,
    )?;

    let y_yoshida = py_matrix(y_yoshida);
    let y_rk4 = py_matrix(y_rk4);
    let y_dp45 = py_matrix(y_dp45);

    let mut df = DataFrame::new(vec![]);
    df.push("t_yoshida", Series::new(t_yoshida));
    df.push("x_yoshida", Series::new(y_yoshida.col(0)));
    df.push("y_yoshida", Series::new(y_yoshida.col(1)));
    df.push("z_yoshida", Series::new(y_yoshida.col(2)));
    df.push("vx_yoshida", Series::new(y_yoshida.col(3)));
    df.push("vy_yoshida", Series::new(y_yoshida.col(4)));
    df.push("vz_yoshida", Series::new(y_yoshida.col(5)));
    df.print();
    df.write_parquet("data_yoshida.parquet", CompressionOptions::Uncompressed)?;

    let mut df = DataFrame::new(vec![]);
    df.push("t_rk4", Series::new(t_rk4));
    df.push("x_rk4", Series::new(y_rk4.col(0)));
    df.push("y_rk4", Series::new(y_rk4.col(1)));
    df.push("z_rk4", Series::new(y_rk4.col(2)));
    df.push("vx_rk4", Series::new(y_rk4.col(3)));
    df.push("vy_rk4", Series::new(y_rk4.col(4)));
    df.push("vz_rk4", Series::new(y_rk4.col(5)));
    df.print();
    df.write_parquet("data_rk4.parquet", CompressionOptions::Uncompressed)?;

    let mut df = DataFrame::new(vec![]);
    df.push("t_dp45", Series::new(t_dp45));
    df.push("x_dp45", Series::new(y_dp45.col(0)));
    df.push("y_dp45", Series::new(y_dp45.col(1)));
    df.push("z_dp45", Series::new(y_dp45.col(2)));
    df.push("vx_dp45", Series::new(y_dp45.col(3)));
    df.push("vy_dp45", Series::new(y_dp45.col(4)));
    df.push("vz_dp45", Series::new(y_dp45.col(5)));
    df.print();
    df.write_parquet("data_dp45.parquet", CompressionOptions::Uncompressed)?;

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct KeplerProblem;

impl KeplerProblem {
    pub fn calc_deriv(&self, state: &State) -> Vec<f64> {
        let r = state.r();
        let r3 = r.powi(3);

        vec![
            state.vx,
            state.vy,
            state.vz,
            -MU * state.x / r3,
            -MU * state.y / r3,
            -MU * state.z / r3,
        ]
    }
}

impl ODEProblem for KeplerProblem {
    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        let state = State::from(y.to_vec());
        let r = state.r();
        let r3 = r.powi(3);

        dy[0] = state.vx;
        dy[1] = state.vy;
        dy[2] = state.vz;
        dy[3] = -MU * state.x / r3;
        dy[4] = -MU * state.y / r3;
        dy[5] = -MU * state.z / r3;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

impl State {
    pub fn r(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}

impl From<Vec<f64>> for State {
    fn from(v: Vec<f64>) -> Self {
        State {
            x: v[0],
            y: v[1],
            z: v[2],
            vx: v[3],
            vy: v[4],
            vz: v[5],
        }
    }
}

impl From<State> for Vec<f64> {
    fn from(s: State) -> Self {
        vec![s.x, s.y, s.z, s.vx, s.vy, s.vz]
    }
}

pub enum OrbitType {
    LEO,
    GEO,
    Molniya,
}

impl ToString for OrbitType {
    fn to_string(&self) -> String {
        match self {
            OrbitType::LEO => "LEO",
            OrbitType::GEO => "GEO",
            OrbitType::Molniya => "Molniya",
        }.to_string()
    }
}

impl OrbitType {
    fn create_orbit(&self) -> Orbit {
        match self {
            OrbitType::LEO => Orbit {
                a: R_EARTH + 500.0,
                e: 0.0,
                i: 0.0,
                raan: 0.0,
                w: 0.0,
                ta: 0.0,
            },
            OrbitType::GEO => Orbit {
                a: R_EARTH + 35786.0,
                e: 0.0,
                i: 0.0,
                raan: 0.0,
                w: 0.0,
                ta: 0.0,
            },
            OrbitType::Molniya => Orbit {
                a: R_EARTH + 26600.0,
                e: 0.74,
                i: 63.4f64.to_radians(),
                raan: 0.0,
                w: 270.0f64.to_radians(),
                ta: 0.0,
            }
        }
    }
}

pub struct Orbit {
    pub a: f64,     // Semi-major axis
    pub e: f64,     // Eccentricity
    pub i: f64,     // Inclination
    pub raan: f64,  // Right ascension of ascending node
    pub w: f64,     // Argument of perigee
    pub ta: f64,    // True anomaly
}

impl Orbit {
    pub fn r(&self) -> f64 {
        self.a * (1.0 - self.e.powi(2)) / (1.0 + self.e * self.ta.cos())
    }

    #[allow(non_snake_case)]
    pub fn initial_state(&self) -> State {
        let r_pf = vec![
            self.r() * self.ta.cos(),
            self.r() * self.ta.sin(),
            0f64
        ];

        let p_orbit = self.a * (1.0 - self.e.powi(2));
        let v_pf = vec![
            - (MU / p_orbit).sqrt() * self.ta.sin(),
            (MU / p_orbit).sqrt() * (self.e + self.ta.cos()),
            0f64
        ];

        let Q = perifocal_to_eci_matrix(&self);
        let r_eci = &Q * &r_pf;
        let v_eci = &Q * &v_pf;

        State {
            x: r_eci[0],
            y: r_eci[1],
            z: r_eci[2],
            vx: v_eci[0],
            vy: v_eci[1],
            vz: v_eci[2],
        }
    }
}

pub fn rot_x(theta: f64) -> Matrix {
    let (s, c) = theta.sin_cos();
    matrix(vec![
        1f64, 0f64, 0f64,
        0f64, c, -s,
        0f64, s, c
    ], 3, 3, Row)
}

pub fn rot_z(theta: f64) -> Matrix {
    let (s, c) = theta.sin_cos();
    matrix(vec![
        c, -s, 0f64,
        s, c, 0f64,
        0f64, 0f64, 1f64
    ], 3, 3, Row)
}

#[allow(non_snake_case)]
pub fn perifocal_to_eci_matrix(orbit: &Orbit) -> Matrix {
    let i = orbit.i;
    let raan = orbit.raan;
    let w = orbit.w;

    let R3_w = rot_z(w);
    let R1_i = rot_x(i);
    let R3_raan = rot_z(raan);

    R3_raan * R1_i * R3_w
}

#[allow(unused)]
pub struct YoshidaSolver {
    problem: KeplerProblem,
}

const W0: f64 = -1.7024143839193153;
const W1: f64 = 1.3512071919596578;

impl YoshidaSolver {
    pub fn new(problem: KeplerProblem) -> Self {
        YoshidaSolver { problem }
    }

    #[allow(non_snake_case)]
    pub fn solve(
        &self,
        t_span: (f64, f64),
        dt: f64,
        initial_condition: &[f64],
    ) -> anyhow::Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let t_vec = linspace(
            t_span.0,
            t_span.1,
            ((t_span.1 - t_span.0) / dt) as usize + 1,
        );
        let state_dim = initial_condition.len();
        let pos_dim = state_dim / 2;
        let pos_init = &initial_condition[..pos_dim];
        let vel_init = &initial_condition[pos_dim..];

        let mut state_vec = vec![vec![0f64; state_dim]; t_vec.len()];
        state_vec[0] = initial_condition.to_vec();

        let c: [f64; 4] = [W1 / 2.0, (W0 + W1) / 2.0, (W0 + W1) / 2.0, W1 / 2.0];
        let d: [f64; 3] = [W1, W0, W1];

        let mut q = pos_init.to_vec();
        let mut p = vel_init.to_vec();

        let acceleration = |q: &[f64]| -> Vec<f64> {
            let r = q.iter().map(|&qi| qi * qi).sum::<f64>().sqrt();
            let r3 = r * r * r;
            q.iter().map(|&qi| -MU * qi / r3).collect::<Vec<f64>>()
        };

        for i in 1..t_vec.len() {
            // Step 1: Drift with c[0]
            for k in 0..pos_dim {
                q[k] += c[0] * dt * p[k];
            }
            // Step 2: Kick with d[0]
            let a = acceleration(&q);
            for k in 0..pos_dim {
                p[k] += d[0] * dt * a[k];
            }
            // Step 3: Drift with c[1]
            for k in 0..pos_dim {
                q[k] += c[1] * dt * p[k];
            }
            // Step 4: Kick with d[1]
            let a = acceleration(&q);
            for k in 0..pos_dim {
                p[k] += d[1] * dt * a[k];
            }
            // Step 5: Drift with c[2]
            for k in 0..pos_dim {
                q[k] += c[2] * dt * p[k];
            }
            // Step 6: Kick with d[2]
            let a = acceleration(&q);
            for k in 0..pos_dim {
                p[k] += d[2] * dt * a[k];
            }
            // Step 7: Drift with c[3]
            for k in 0..pos_dim {
                q[k] += c[3] * dt * p[k];
            }

            state_vec[i] = concat(&q, &p);
        }

        Ok((t_vec, state_vec))
    }
}
