use std::sync::Arc;

use peroxide::fuga::*;
use anise::{
    almanac::metaload::MetaFile,
    constants::{
        celestial_objects::EARTH,
        frames::{EARTH_J2000, IAU_EARTH_FRAME}
    },
};
use hifitime::{Epoch, TimeUnits, Unit};
use nyx_space as nyx;
use nyx::{
    cosmic::{eclipse::EclipseLocator, GuidanceMode, Mass, MetaAlmanac, Orbit, SRPData},
    dynamics::{
        OrbitalDynamics, SpacecraftDynamics, ConstantDrag,
    },
    io::ExportCfg,
    mc::{MonteCarlo, MvnSpacecraft},
    propagators::{Propagator, ErrorControl, IntegratorOptions},
    Spacecraft, State,
};
use pretty_env_logger as pel;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    pel::init();

    let almanac = Arc::new(MetaAlmanac::latest().map_err(Box::new)?);
    let epoch = Epoch::from_gregorian_utc_hms(2025, 2, 6, 18, 25, 0);

    let earth_j2000 = almanac.frame_from_uid(EARTH_J2000)?;
    let molniya = Orbit::try_keplerian_altitude(
        26600.0,
        0.74,
        63.4,
        0.0,
        270.0,
        0.0,
        epoch,
        earth_j2000,
    )?;

    let sc = Spacecraft::builder()
        .orbit(molniya)
        .mass(Mass::from_dry_mass(1000f64))
        .build();

    let orbital_dynamics = OrbitalDynamics::point_masses(vec![EARTH]);
    let no_dyn = ConstantDrag {
        rho: 0f64,
        drag_frame: EARTH_J2000,
        estimate: false,
    };
    let arc_no_dyn = Arc::new(no_dyn);

    let sc_dynamics = SpacecraftDynamics::from_model(orbital_dynamics, arc_no_dyn);

    let prop_time = 1000.0 * Unit::Day;

    let setup = Propagator::rk89(
        sc_dynamics.clone(),
        IntegratorOptions::builder()
            .min_step(1.0_f64.seconds())
            .error_ctrl(ErrorControl::RSSCartesianStep)
            .build(),
    );

    let mc_rv = MvnSpacecraft::new(
        sc,
        vec![],
    )?;
    
    let my_mc = MonteCarlo::new(
        sc,
        mc_rv,
        "test".to_string(),
        Some(42),
    );

    let num_runs = 1;
    let results = my_mc.run_until_epoch(setup, almanac.clone(), sc.epoch() + prop_time, num_runs);

    results.to_parquet("test.parquet", None, ExportCfg::default(), almanac)?;

    

    Ok(())
}
