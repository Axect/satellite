use peroxide::fuga::*;
use dialoguer::Select;
use std::time::Instant;

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

    let selected_orbit_type = orbits[selection];
    let selected_orbit = selected_orbit_type.create_orbit();
    let initial_state = selected_orbit.initial_state();

    let perturbations = vec![
        "None",
        "J2",
    ];
    let selection = Select::new()
        .with_prompt("Select perturbation")
        .items(&perturbations)
        .default(0)
        .interact()
        .unwrap();
    let is_perturbed = selection == 1;

    let t0 = 0.0;
    let tf = 86400.0 * 100000.0;
    let dt = 60f64;

    let problem = KeplerProblem {
        is_perturbed,
    };
    let yoshida_solver = YoshidaSolver::new(problem);
    let rk4 = RK4;
    let dp45 = DP45 {
        max_step_iter: 1000,
        max_step_size: 2.0 * dt,
        min_step_size: 1e-3 * dt,
        safety_factor: 0.9,
        tol: 1e-1,
    };
    let gl4 = GL4 {
        solver: ImplicitSolver::Broyden,
        tol: 1e-12,
        max_step_iter: 100,
    };

    let rk4_solver = CompactODESolver::new(rk4);
    let dp45_solver = CompactODESolver::new(dp45);
    let gl4_solver = CompactODESolver::new(gl4);

    let y0 = Vec::from(initial_state);

    let yoshida_start = Instant::now();
    let (t_yoshida, y_yoshida) = yoshida_solver.solve(
        (t0, tf),
        dt,
        &y0,
    )?;
    let yoshida_elapsed = yoshida_start.elapsed();
    println!("Yoshida elapsed: {:?}", yoshida_elapsed);
    let y_yoshida = py_matrix(y_yoshida);
    save_data(t_yoshida, y_yoshida, "yoshida", is_perturbed)?;

    let rk4_start = Instant::now();
    let (t_rk4, y_rk4) = rk4_solver.solve(
        &problem,
        (t0, tf),
        dt,
        &y0,
    )?;
    let rk4_elapsed = rk4_start.elapsed();
    println!("RK4 elapsed: {:?}", rk4_elapsed);
    let y_rk4 = py_matrix(y_rk4);
    save_data(t_rk4, y_rk4, "rk4", is_perturbed)?;

    let dp45_start = Instant::now();
    let (t_dp45, y_dp45) = dp45_solver.solve(
        &problem,
        (t0, tf),
        dt,
        &y0,
    )?;
    let dp45_elapsed = dp45_start.elapsed();
    println!("DP45 elapsed: {:?}", dp45_elapsed);
    let y_dp45 = py_matrix(y_dp45);
    save_data(t_dp45, y_dp45, "dp45", is_perturbed)?;

    let gl4_start = Instant::now();
    let (t_gl4, y_gl4) = gl4_solver.solve(
        &problem,
        (t0, tf),
        dt,
        &y0,
    )?;
    let gl4_elapsed = gl4_start.elapsed();
    println!("GL4 elapsed: {:?}", gl4_elapsed);
    let y_gl4 = py_matrix(y_gl4);
    save_data(t_gl4, y_gl4, "gl4", is_perturbed)?;

    Ok(())
}

fn save_data(t: Vec<f64>, result: Matrix, model: &str, is_perturbed: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut df = DataFrame::new(vec![]);
    df.push("t", Series::new(t));
    df.push("x", Series::new(result.col(0)));
    df.push("y", Series::new(result.col(1)));
    df.push("z", Series::new(result.col(2)));
    df.push("vx", Series::new(result.col(3)));
    df.push("vy", Series::new(result.col(4)));
    df.push("vz", Series::new(result.col(5)));
    df.print();
    let filename = format!("data_{}_{}.parquet", model, if is_perturbed { "J2" } else { "2BD" });
    df.write_parquet(&filename, CompressionOptions::Uncompressed)
}

#[derive(Debug, Clone, Copy)]
pub struct KeplerProblem {
    is_perturbed: bool,
}

impl KeplerProblem {
    pub fn calc_deriv(&self, state: &State) -> Vec<f64> {
        let r = state.r();
        let r2 = r * r;
        let r3 = r.powi(3);
        let r5 = r.powi(5);

        let (j2_x, j2_y, j2_z) = if self.is_perturbed {
            let factor = 1.5 * J2 * MU * R_EARTH * R_EARTH / r5;
            (
                factor * state.x * (5.0 * state.z * state.z / r2 - 1.0),
                factor * state.y * (5.0 * state.z * state.z / r2 - 1.0),
                factor * state.z * (5.0 * state.z * state.z / r2 - 3.0)
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        vec![
            state.vx,
            state.vy,
            state.vz,
            -MU * state.x / r3 + j2_x,
            -MU * state.y / r3 + j2_y,
            -MU * state.z / r3 + j2_z,
        ]
    }
}

impl ODEProblem for KeplerProblem {
    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        let state = State::from(y.to_vec());

        let deriv = self.calc_deriv(&state);
        dy.copy_from_slice(&deriv);

        Ok(())
    }
}

pub struct CompactODESolver<I: ODEIntegrator> {
    integrator: I,
}

impl<I: ODEIntegrator> CompactODESolver<I> {
    pub fn new(integrator: I) -> Self {
        CompactODESolver { integrator }
    }
}

impl<I: ODEIntegrator> ODESolver for CompactODESolver<I> {
    fn solve<P: ODEProblem>(
            &self,
            problem: &P,
            t_span: (f64, f64),
            dt: f64,
            initial_conditions: &[f64],
        ) -> anyhow::Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let mut t = t_span.0;
        let mut dt = dt;
        let mut y = initial_conditions.to_vec();
        let mut t_vec = vec![t];
        let mut y_vec = vec![y.clone()];

        let mut count = 1usize;
        while t < t_span.1 {
            let dt_step = self.integrator.step(problem, t, &mut y, dt)?;
            t += dt;
            dt = dt_step;

            if count % 100 == 0 {
                t_vec.push(t);
                y_vec.push(y.clone());
            }
            count += 1;
        }

        Ok((t_vec, y_vec))
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

#[derive(Debug, Clone, Copy)]
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
        let total_steps = ((t_span.1 - t_span.0) / dt) as usize + 1;

        let state_dim = initial_condition.len();
        let pos_dim = state_dim / 2;
        let pos_init = &initial_condition[..pos_dim];
        let vel_init = &initial_condition[pos_dim..];

        let mut t_vec = vec![t_span.0];
        let mut state_vec = vec![initial_condition.to_vec()];

        let c: [f64; 4] = [W1 / 2.0, (W0 + W1) / 2.0, (W0 + W1) / 2.0, W1 / 2.0];
        let d: [f64; 3] = [W1, W0, W1];

        let mut q = pos_init.to_vec();
        let mut p = vel_init.to_vec();

        let acceleration = |q: &[f64], p: &[f64]| {
            let state = State {
                x: q[0],
                y: q[1],
                z: q[2],
                vx: p[0],
                vy: p[1],
                vz: p[2],
            };
            self.problem.calc_deriv(&state)[pos_dim..].to_vec()
        };

        for i in 1..total_steps {
            // Step 1: Drift with c[0]
            for k in 0..pos_dim {
                q[k] += c[0] * dt * p[k];
            }
            // Step 2: Kick with d[0]
            let a = acceleration(&q, &p);
            for k in 0..pos_dim {
                p[k] += d[0] * dt * a[k];
            }
            // Step 3: Drift with c[1]
            for k in 0..pos_dim {
                q[k] += c[1] * dt * p[k];
            }
            // Step 4: Kick with d[1]
            let a = acceleration(&q, &p);
            for k in 0..pos_dim {
                p[k] += d[1] * dt * a[k];
            }
            // Step 5: Drift with c[2]
            for k in 0..pos_dim {
                q[k] += c[2] * dt * p[k];
            }
            // Step 6: Kick with d[2]
            let a = acceleration(&q, &p);
            for k in 0..pos_dim {
                p[k] += d[2] * dt * a[k];
            }
            // Step 7: Drift with c[3]
            for k in 0..pos_dim {
                q[k] += c[3] * dt * p[k];
            }

            if i % 100 == 0 {
                t_vec.push(t_span.0 + i as f64 * dt);
                state_vec.push(concat(&q, &p));
            }
        }
        Ok((t_vec, state_vec))
    }
}
