use black_splat::config::Config;

mod example_game;

use example_game::MujocoTestGame;

fn main() {
    let config_file_text = include_str!("game_config.txt");
    let game_config = Config::new(config_file_text);

    let run_game = black_splat::run_game::<MujocoTestGame>(game_config);

    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(run_game);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(run_game);
    }
}
