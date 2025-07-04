#include <SFML/Config.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics.hpp>
#include <strings.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <array>
#include <atomic>
#include <xmmintrin.h>
#include <cmath>
#include <cassert>


const float pi = 3.141592653589f;
const float pi2 = 6.28318530718f;
const float pio2 = pi*0.5;
const float pi_inverse = 1/pi;
const float pi2_inverse = 1/pi2;
const float rs = 1.0f;
const float risco = 2.0f;
const float rlast = 4.0f;
const float tff_inverse = 0.003921569;
const float tpiof = 3*pi*0.25;
const float r_pion2_inverse = 0.797884561;

thread_local float gamma001;
thread_local float gamma010;
thread_local float gamma100;
thread_local float gamma111;
thread_local float gamma122;
thread_local float gamma133;
thread_local float gamma212;
thread_local float gamma233;
thread_local float gamma221;
thread_local float gamma313;
thread_local float gamma323;
thread_local float gamma331;
thread_local float gamma332;

float tetrad_00 = 0;
float tetrad_11 = 0;
float tetrad_22 = 0;
float tetrad_33 = 0;

float tetrad_inverse_00 = 0;
float tetrad_inverse_11 = 0;
float tetrad_inverse_22 = 0;
float tetrad_inverse_33 = 0;


sf::Image accretion_disk_texture;


struct Vector3f {
    float x, y, z;

    Vector3f() : x(0), y(0), z(0) {}
    Vector3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    float& operator()(int i) { assert(i >= 0 && i < 3); return (&x)[i]; }
    const float& operator()(int i) const { assert(i >= 0 && i < 3); return (&x)[i]; }

    float& operator[](int i) { return operator()(i); }
    const float& operator[](int i) const { return operator()(i); }

    void setZero() { x = y = z = 0.0f; }

    Vector3f operator+(const Vector3f& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vector3f operator-(const Vector3f& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vector3f operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    friend Vector3f operator*(float scalar, const Vector3f& v) { return v * scalar; }

    float dot(const Vector3f& other) const { return x * other.x + y * other.y + z * other.z; }
    float squaredNorm() const { return dot(*this); }
    float norm() const { return std::sqrt(squaredNorm()); }

    Vector3f normalized() const {
        float n = norm();
        return (n > 0) ? (*this) * (1.0f / n) : Vector3f(0, 0, 0);
    }
};

struct Vector4f {
    float x, y, z, w;

    Vector4f() : x(0), y(0), z(0), w(0) {}
    Vector4f(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    float& operator()(int i) { assert(i >= 0 && i < 4); return (&x)[i]; }
    const float& operator()(int i) const { assert(i >= 0 && i < 4); return (&x)[i]; }

    float& operator[](int i) { return operator()(i); }
    const float& operator[](int i) const { return operator()(i); }

    void setZero() { x = y = z = w = 0.0f; }

    Vector4f operator+(const Vector4f& other) const { return {x + other.x, y + other.y, z + other.z, w + other.w}; }
    Vector4f operator-(const Vector4f& other) const { return {x - other.x, y - other.y, z - other.z, w - other.w}; }
    Vector4f operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar, w * scalar}; }
    friend Vector4f operator*(float scalar, const Vector4f& v) { return v * scalar; }


    float dot(const Vector4f& other) const { return x * other.x + y * other.y + z * other.z + w * other.w; }
    float squaredNorm() const { return dot(*this); }
    float norm() const { return std::sqrt(squaredNorm()); }

    Vector4f normalized() const {
        float n = norm();
        return (n > 0) ? (*this) * (1.0f / n) : Vector4f(0, 0, 0, 0);
    }
};



std::vector<std::vector<float>> gaussian_kernel;


inline float clamp_(float value , float min , float max){
    if(value < min)return min;
    if(value > max)return max;
    return value;
}


__attribute__((always_inline))inline float fast_sqrt(float x){
    __m128 v = _mm_set_ss(x);
    __m128 r = _mm_rsqrt_ss(v);
    __m128 half = _mm_set_ss(0.5);
    __m128 three = _mm_set_ss(1.5);
    __m128 r2 = _mm_mul_ss(r,r);
    __m128 xr2 = _mm_mul_ss(r2,v);
    __m128 sub = _mm_sub_ss(three,_mm_mul_ss(half , xr2));
    r = _mm_mul_ss(r , sub);
    __m128 s = _mm_mul_ss(r , v);
    return _mm_cvtss_f32(s);

}


inline float Q_rsqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5F;
    x2 = number * 0.5F;
    y = number;
    i = *(long*)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;
    return y * (threehalfs - (x2 * y * y));
}

using image = sf::Image;


inline void Christoffel_symbol_of2(const Vector4f& position) {
    float r = position(1);
    float theta = position(2);
    float r_inverse = 1.0f / r;
    float rmrs = r - rs;
    float rmrs_inverse = 1.0f / rmrs;
    float sint = sinf(theta);
    float cost = cosf(theta);
    float cott = cost / sint;
    float rirmi = r_inverse * rmrs_inverse;

    gamma001 = gamma010 = rs * 0.5f * rirmi;
    gamma100 = rs * rmrs * 0.5f * r_inverse * r_inverse * r_inverse;
    gamma111 = -rs * 0.5f * rirmi;
    gamma122 = -rmrs;
    gamma133 = -rmrs * sint * sint;
    gamma212 = gamma221 = r_inverse;
    gamma233 = -sint * cost;
    gamma313 = gamma331 = r_inverse;
    gamma323 = gamma332 = cott;

}

inline float get_glow(float& position){
    if(position < 1.0) return 0;
    float glow = 1/position;
    return glow*glow;
}

void set_schwarzschild_tetrad(const Vector4f& camera_position) {
    float r = camera_position(1);
    float rsor = rs/r;
    float k = fast_sqrt(1.0f - rsor);
    float r_inverse = 1.0f / r;
    float theta = camera_position(2);
    float ook = Q_rsqrt(1.0f - rsor);


    tetrad_00 = ook;
    tetrad_11 = k;
    tetrad_22 = r_inverse;
    tetrad_33 = r_inverse / sinf(theta);

    tetrad_inverse_00 = k;
    tetrad_inverse_11 = ook;
    tetrad_inverse_22 = r;
    tetrad_inverse_33 = r*sinf(theta);

}


inline Vector3f get_pixel_direction(int x, int y, int fov, int screen_width, int screen_height) {
    fov = 90;
    float fov_rad = (fov * 0.002778f) * pi2;
    float screen_dist = screen_width * 0.5f;
    Vector3f dir;
    dir.x = x - screen_width * 0.5f;
    dir.y = y - screen_height * 0.5f;
    dir.z = screen_dist;
    float mag = Q_rsqrt(dir.x*dir.x +dir.y*dir.y +dir.z*dir.z);
    
    dir.x = dir.x*mag;
    dir.y = dir.y*mag;
    dir.z = dir.z*mag;

    return dir;
}


struct geodesic {
    Vector4f position;
    Vector4f velocity;
};


inline geodesic get_lightlike_geodesic(const Vector4f& position, const Vector3f& direction) {
    geodesic g;
    g.position = position;
    g.velocity.setZero();
    g.velocity(0) = -tetrad_00;
    g.velocity(1) = direction.x * tetrad_11;
    g.velocity(2) = direction.y * tetrad_22;
    g.velocity(3) = direction.z * tetrad_33;
    return g;
}

inline Vector4f get_acceleration(const Vector4f& position, const Vector4f& v) {
    Christoffel_symbol_of2(position);
    Vector4f acc = {0,0,0,0};
    acc(0) = -(gamma001*v(0)*v(1));
    acc(1) = -(gamma100*v(0)*v(0) + gamma111*v(1)*v(1) + gamma122*v(2)*v(2) + gamma133*v(3)*v(3));
    acc(2) = -(gamma212*v(1)*v(2) + gamma233*v(3)*v(3) + gamma221*v(2)*v(1));
    acc(3) = -(gamma313*v(1)*v(3) + gamma323*v(2)*v(3) + gamma331*v(3)*v(1) + gamma332*v(3)*v(2));
    return acc;
}

inline Vector4f get_acceleration_2(const Vector4f& pos, const Vector4f& vel) {
    float r = pos(1);
    float theta = pos(2);
    float sint = sinf(theta);
    float cost = cosf(theta);
    float cott = cost / sint;
    float r_inv = 1.0f / r;
    float rmrs = r - rs;
    float rmrs_inv = 1.0f / rmrs;
    float rirmi = r_inv * rmrs_inv;

    Vector4f acc;
    acc.setZero();

    acc(0) = -rs * rirmi * vel(0) * vel(1);

    acc(1) =
        -rs * rmrs * 0.5f * vel(0) * vel(0) / (r * r * r)
        + rs * 0.5f * rirmi * vel(1) * vel(1)
        + rmrs * vel(2) * vel(2)
        + rmrs * sint * sint * vel(3) * vel(3);

    acc(2) =
        -2 * r_inv * vel(1) * vel(2)
        + sint * cost * vel(3) * vel(3);

    acc(3) =
        -2 * r_inv * vel(1) * vel(3)
        - 2 * cott * vel(2) * vel(3);

    return acc;
}


inline Vector3f get_local_direction(const Vector4f& velocity , const Vector4f& camera_position){
    Vector3f local_dir;
    local_dir(2) = -velocity(1) * tetrad_inverse_11;
    local_dir(1) = velocity(2) * tetrad_inverse_22;
    local_dir(0) = velocity(3) * tetrad_inverse_33;

    float mag = Q_rsqrt(local_dir(0)*local_dir(0) + local_dir(1)*local_dir(1) +local_dir(2)*local_dir(2));


    local_dir(0) = local_dir(0)*mag;
    local_dir(1) = local_dir(1)*mag;
    local_dir(2) = local_dir(2)*mag;

    return local_dir;
}



enum integration_result {
    UNKNOWN,
    ESCAPED,
    EVENTHORIZON
};


Vector3f render_pixel(int x, int y, int screen_width, int screen_height, const Vector4f& camera_position, const sf::Image& background) {
    Vector3f dir = get_pixel_direction(x, y, 90, screen_width, screen_height);
    float phi_ = 0;
    Vector3f mod_dir{-dir.z, dir.y, dir.x};
    geodesic g = get_lightlike_geodesic(camera_position, mod_dir);
    float dt = 0.002f;
    float glow = 0;
    float r_;
    float g_;
    float b_;
    for (int i = 0; i < 10000; ++i) {
        Vector4f acc = get_acceleration(g.position, g.velocity);
        acc = acc*dt;
        //acc(1) = acc(1)*dt;
        //acc(2) = acc(2)*dt;
        //acc(3) = acc(3)*dt;

        g.velocity = g.velocity + acc;
        //g.velocity(1) = g.velocity(1) + acc(1);
        //g.velocity(2) = g.velocity(2) + acc(2);
        //g.velocity(3) = g.velocity(3) + acc(3);


        g.position = g.position + g.velocity * dt;
        //g.position(1) = g.position(1) + g.velocity(1) * dt;
        //g.position(2) = g.position(2) + g.velocity(2) * dt;
        //g.position(3) = g.position(3) + g.velocity(3) * dt;

        float position = g.position(1);
        position = std::fabs(position);
        

        glow = get_glow(position);
        glow = clamp_(glow , 0 , 0.5);

        if (position > 5.0f) {
            Vector3f local_dir = get_local_direction(g.velocity, g.position);
            float theta = acosf(local_dir(1));
            phi_ = atan2(local_dir(2), local_dir(0)) + pi;
            float u = phi_ * pi2_inverse;
            float v = theta * pi_inverse;


            int tx = (int)(u * 1080);
            if (tx < 0) tx =tx + 1080;
            else if (tx >= 1080) tx = tx - 1080;

            int ty = (int)(v * 720);
            if (ty < 0) ty = ty + 720;
            else if (ty >= 720) ty = ty - 720;


            sf::Color color = background.getPixel(tx, ty);

            glow = clamp_(glow * 2.0f, 0.0f, 1.0f);

            r_ = (1.0f - glow) * (color.r * tff_inverse) + glow * 1.0f;
            g_ = (1.0f - glow) * (color.g * tff_inverse) + glow * 1.0f;
            b_ = (1.0f - glow) * (color.b * tff_inverse) + glow * 1.0f;
            
            return Vector3f(r_ , g_ , b_);

        }
        if(g.position(2) < pio2 + 0.001 && g.position(2) > pio2 - 0.001 ){
            phi_ = g.position(3);
            if(position < rlast && position > risco){
                position = position - 1;
                position = position*133.33;
                float x = cosf(phi_)*position;
                float y = sinf(phi_)*position;
                sf::Color c = accretion_disk_texture.getPixel(x + 400, y + 400);
                glow = clamp_(glow * 2.0f, 0.0f, 1.0f);

                r_ = (1.0f - glow) * (c.r * tff_inverse) + glow * 1.0f;
                g_ = (1.0f - glow) * (c.g * tff_inverse) + glow * 1.0f;
                b_ = (1.0f - glow) * (c.b * tff_inverse) + glow * 1.0f;
            
                return Vector3f(r_ , g_ , b_);
            }
        }
        if (position < rs + 0.0001f){ 
            return Vector3f(0.f, 0.f, 0.f);
        }
    }
    return Vector3f(0.f, 0.f, 0.f);
}

std::vector<Vector3f> get_pixels(int screen_width, int screen_height, const Vector4f& camera_position, const sf::Image& background) {
    std::vector<Vector3f> result(screen_width * screen_height);
    std::atomic_int next_pixel{0};
    int total_pixels = screen_width * screen_height;
    int thread_count = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (int t = 0; t < thread_count; ++t) {
        threads.emplace_back([&]() {
            while (true) {
                int start = next_pixel.fetch_add(64);
                if (start >= total_pixels) break;
                for (int i = start; i < std::min(start + 64, total_pixels); ++i) {
                    int x = i % screen_width;
                    int y = i / screen_width;
                    result[i] = render_pixel(x, y, screen_width, screen_height, camera_position, background);
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

void set_gaussian_kernel(){
    gaussian_kernel.resize(9, std::vector<float>(9)); // Allocate 9 rows, each with 9 columns
    for(int y = 0 ; y < 9 ; y++){
        for(int x = 0 ; x < 9 ; x++){
            float dist_squared = (x - 4)*(x - 4) + (y - 4)*(y - 4);
            gaussian_kernel[y][x] = r_pion2_inverse*exp(-2*dist_squared);
        }
    }
}

void apply_gaussian_blur(std::vector<Vector3f>& bloom , std::vector<Vector3f>& gaussian_blur){
    for(int y = 0 ; y < 720 ; y++){
        for(int x = 0 ; x < 1080 ; x++){
            float sum_r = 0;
            float sum_g = 0;
            float sum_b = 0;
            for(int i = -4 ; i < 5 ; i++){
                for(int j = -4 ; j < 5 ; j++){
                    //i = y , j = x
                    sum_r += gaussian_kernel[i][j]*bloom[(y + i)*1080 + x + j].x;
                    sum_g += gaussian_kernel[i][j]*bloom[(y + i)*1080 + x + j].y;
                    sum_b += gaussian_kernel[i][j]*bloom[(y + i)*1080 + x + j].z;


                }
            }
            gaussian_blur[y*1080 + x].x = sum_r;
            gaussian_blur[y*1080 + x].y = sum_g;
            gaussian_blur[y*1080 + x].z = sum_b;


        }
    }
}



int main() {
    set_gaussian_kernel();
    accretion_disk_texture.loadFromFile("texture.png");
    Vector4f camera_pos{0.f, 5.0f, pi * 0.472f, -pi * 0.5f};
    int screen_width = 1080;
    int screen_height = 720;

    set_schwarzschild_tetrad(camera_pos);

    sf::Image input;
    input.loadFromFile("image.png");

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Vector3f> pixels = get_pixels(screen_width, screen_height, camera_pos, input);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::vector<Vector3f> bloom(screen_width*screen_height);
    std::vector<Vector3f> gaussian_blur;

    for(int i = 0 ; i < screen_height*screen_width ; i++){
        Vector3f color = pixels[i];
        float brightness = color.dot(Vector3f(0.2126f , 0.7152f , 0.07227f));
        bloom[i] = (brightness > 0.7f) ? color : Vector3f(0,0,0);
    }

    apply_gaussian_blur(bloom , gaussian_blur);

    for(int i = 0 ; i < 1080*720 ; i++){
        pixels[i].x = gaussian_blur[i].x*pixels[i].x;
        pixels[i].y = gaussian_blur[i].y*pixels[i].y;
        pixels[i].z = gaussian_blur[i].z*pixels[i].z;
    }


    sf::Image output;
    output.create(screen_width, screen_height);

    for (int y = 0; y < screen_height; ++y) {
        for (int x = 0; x < screen_width; ++x) {
            const Vector3f& color = pixels[y * screen_width + x];
            output.setPixel(x, y, sf::Color(color.x * 255, color.y * 255, color.z * 255));
        }
    }

    output.saveToFile("render.png");

    std::cout << "Render Time (sec): "<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "\n";

    return 0;
}
    
