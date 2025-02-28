extends Node2D

func _ready():
    var material = ShaderMaterial.new()
    material.shader = load("res://shaders/super_resolution.shader")
    
    # Cargar kernels desde un archivo JSON
    var kernels = load_kernels("res://data/kernels.json")
    material.set_shader_param("kernels", kernels)
    
    # Aplicar a un Viewport
    $Viewport.material = material

func load_kernels(path: String) -> Array:
    var file = File.new()
    file.open(path, File.READ)
    var data = JSON.parse(file.get_as_text()).result
    file.close()
    return data["kernels"]