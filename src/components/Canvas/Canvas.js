import eventBus from "../../eventbuss";
import * as tf from '@tensorflow/tfjs';
export default {
    name: "gan-generator",
    data: () => ({
        modelsInfo: {
            dcgan64: {
                description: 'DCGAN, 64x64 (16 MB)',
                model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-dcgan-celebahq-64/tfjs_SmoothedGenerator_50000/model.json",
                model_size: 64,
                model_latent_dim: 128,
                draw_multiplier: 8,
                animate_frame: 200,
            },
            resnet128: {
                description: 'ResNet, 128x128 (252 MB)',
                model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-resent128-celebahq-128/tfjs_SmoothedGenerator_20000/model.json",
                model_size: 128,
                model_latent_dim: 128,
                draw_multiplier: 2,
                animate_frame: 10
            },
            resnet256: {
                description: 'ResNet, 256x256 (252 MB)',
                model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_40000/model.json",
                model_size: 256,
                model_latent_dim: 128,
                draw_multiplier: 1,
                animate_frame: 10
            }
        },
        currentModelName: "dcgan64",
        tfComputingDellay: 2000,
        model_promise_cache: {},
        model_promise: null,
        model_name: null,
        model_description: null,
        start_time: null,
        imageExist: false,
        zoomLevel: 0,
        zoomLevelMin: 0,
        zoomLevelMax: 3,
    }),
    created() {
        eventBus.$on("generate_image", () => {
            this.generateImage();
        });
        eventBus.$on("animate_image", () => {
            this.animateImage();
        });
        eventBus.$on("change_model", (modelName) => {
            this.currentModelName = modelName.toLowerCase().replace("-", "");
            this.setupModel();
        });
    },

    methods: {
        setupModel: function() {
            let model_info = this.modelsInfo[this.currentModelName];
            let model_size = model_info.model_size,
                model_url = model_info.model_url,
                draw_multiplier = model_info.draw_multiplier;

            this.model_description = model_info.description;
            this.computing_prep_canvas(model_size * draw_multiplier);

            if (this.currentModelName in this.model_promise_cache) {
                this.model_promise = this.model_promise_cache[this.currentModelName];
            } else {
                eventBus.$emit("disableControllButtons", true);
                this.model_promise = tf.loadModel(model_url);
                this.model_promise.then((model) => {
                    return this.resolveAfterDellay(model, this.tfComputingDellay);
                }).then(() => {
                    eventBus.$emit("disableControllButtons", false);
                });
                this.model_promise_cache[this.currentModelName] = this.model_promise;
            }
        },
        generateImage: function() {
            const { model_size, draw_multiplier, model_latent_dim } = this.initGeneratorMetaData();
            eventBus.$emit("disableControllButtons", true);
            this.model_promise.then((model) => {
                return this.resolveAfterDellay(model, this.tfComputingDellay);
            }).then((model) => {
                return this.computing_generate_main(model, model_size, draw_multiplier, model_latent_dim);
            }).then(() => {
                eventBus.$emit("disableControllButtons", false);
                this.imageExist = true;
            });

        },
        animateImage: function() {
            const { draw_multiplier, animate_frame } = this.initAnimationMetaData();
            eventBus.$emit("disableControllButtons", true);
            this.model_promise.then((model) => {
                return this.resolveAfterDellay(model, this.tfComputingDellay);
            }).then((model) => {
                return this.computing_animate_latent_space(model, draw_multiplier, animate_frame);
            }).then(() => {
                eventBus.$emit("disableControllButtons", false);
            });
        },
        computing_prep_canvas: function(size) {
            let canvas = document.getElementById("image_canvas");
            let ctx = canvas.getContext("2d");
            ctx.canvas.width = size;
            ctx.canvas.height = size;
        },
        resolveAfterDellay: function(x, ms) {
            return new Promise(resolve => {
                setTimeout(() => {
                    resolve(x);
                }, ms);
            });
        },
        computing_generate_main: async function async(model, size, draw_multiplier, latent_dim) {
            const y = tf.tidy(() => {
                const z = tf.randomNormal([1, latent_dim]);
                const y = model.predict(z).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(0.5));
                return this.enlargeImage(y, draw_multiplier);

            });
            let canvas = document.getElementById("image_canvas");
            var img = new Image();
            eventBus.$emit("set_chart_data", y);
            await tf.toPixels(y, canvas);
            img.onload = function() {
                var canvas = new fabric.Canvas("image_canvas");
                var oldCanvas = new fabric.Image(img, { width: canvas.width / 2, height: canvas.height / 2 });
                canvas.add(oldCanvas);
            }
            img.src = canvas.toDataURL("image/png");
        },
        computing_animate_latent_space: async function(model, draw_multiplier, animate_frame) {
            const inputShape = model.inputs[0].shape.slice(1);
            const shift = tf.randomNormal(inputShape).expandDims(0);
            const freq = tf.randomNormal(inputShape, 0, 0.1).expandDims(0);

            let c = document.getElementById("image_canvas");
            let i = 0;
            while (i < animate_frame) {
                i++;
                const y = tf.tidy(() => {
                    const z = tf.sin(tf.scalar(i).mul(freq).add(shift));
                    const y = model.predict(z).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(0.5));
                    return this.enlargeImage(y, draw_multiplier);
                });
                await tf.toPixels(y, c);
                await tf.nextFrame();
            }
        },
        fixSize_64x64: function() {
            let canvas = document.getElementById("image_canvas");
            var img = new Image();
            let self = this;
            img.onload = function() {
                var canvas = new fabric.Canvas("image_canvas");
                const size = self.modelsInfo[self.currentModelName].model_size;
                var oldCanvas = new fabric.Image(img, { width: size, height: size });
                canvas.add(oldCanvas);
            }
            img.src = canvas.toDataURL("image/png");
        },
        enlargeImage: function(y, draw_multiplier) {
            if (draw_multiplier === 1) {
                return y;
            }
            let size = y.shape[0];
            return y.expandDims(2).tile([1, 1, draw_multiplier, 1]).reshape([size, size * draw_multiplier, 3]).expandDims(1).tile([1, draw_multiplier, 1, 1]).reshape([size * draw_multiplier, size * draw_multiplier, 3])
        },
        initAnimationMetaData: function() {
            let model_info = this.modelsInfo[this.currentModelName];
            let model_size = model_info.model_size,
                draw_multiplier = model_info.draw_multiplier,
                animate_frame = model_info.animate_frame;
            this.computing_prep_canvas(model_size * draw_multiplier);
            return { draw_multiplier, animate_frame };
        },
        initGeneratorMetaData: function() {
            const model_info = this.modelsInfo[this.currentModelName];
            const model_size = model_info.model_size,
                model_latent_dim = model_info.model_latent_dim,
                draw_multiplier = model_info.draw_multiplier;
            this.computing_prep_canvas(model_size * draw_multiplier);
            return { model_size, draw_multiplier, model_latent_dim };
        },
    }
}