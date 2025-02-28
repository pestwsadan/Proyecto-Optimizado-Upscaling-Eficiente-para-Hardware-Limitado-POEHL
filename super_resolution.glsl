precision mediump float;

uniform sampler2D inputTexture;
uniform vec2 inputSize;
uniform mat3 kernels[3];

void main() {
    vec2 uv = gl_FragCoord.xy / inputSize;
    vec3 color = texture2D(inputTexture, uv).rgb;

    // Capa 1: Convolución 5x5 (simplificada a 3x3)
    vec3 layer1 = vec3(0.0);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec2 offset = vec2(float(i), float(j)) / inputSize;
            layer1 += texture2D(inputTexture, uv + offset).rgb * kernels[0][i+1][j+1];
        }
    }
    layer1 = max(layer1, vec3(0.0));

    // Capa 2: Convolución 3x3
    vec3 layer2 = vec3(0.0);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            layer2 += layer1 * kernels[1][i+1][j+1];
        }
    }
    layer2 = max(layer2, vec3(0.0));

    // Capa 3 + PixelShuffle (simplificado)
    gl_FragColor = vec4(layer2, 1.0);
}