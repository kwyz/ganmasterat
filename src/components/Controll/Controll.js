import eventBus from "@/eventbuss.js";
export default {
    name: "controll-component",
    data: () => ({
        items: ["DCGAN-64", "ResNet-128", "ResNet-256"],
        ganModel: "",
        disabled: false,

    }),
    created() {
        eventBus.$on("disableControllButtons", (disabled) => {
            this.disabled = disabled;
        });
    },
    watch: {
        ganModel: function(val) {
            this.disabled = true;
            eventBus.$emit("change_model", val);
        }
    },
    methods: {
        generateImage: () => {
            eventBus.$emit("generate_image");
        },
        animateImage: () => {
            eventBus.$emit("animate_image");
        }
    }
}