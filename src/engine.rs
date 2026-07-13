use crate::{config::*, game_object::*, input::*, renderer::*};

pub trait GameEngine {
    fn new(game_config: &Config) -> Self;

    fn get_game_objects(&self) -> &Vec<GameObject>;

    #[allow(async_fn_in_trait)]
    async fn initialize_world(&mut self, renderer: &mut Renderer<'_>, game_config: &mut Config);

    // Do not override tick_frame().  Put custom code in tick_frame_internal()
    fn tick_frame(
        &mut self,
        renderer: &mut Renderer<'_>,
        input_manager: &mut InputManager,
        game_config: &mut Config,
    ) {
        game_config.update_frame_times();
        self.tick_frame_internal(renderer, input_manager, game_config);
        input_manager.update_key_states();
    }

    fn tick_frame_internal(
        &mut self,
        renderer: &mut Renderer<'_>,
        input_manager: &InputManager,
        game_config: &Config,
    );
}
